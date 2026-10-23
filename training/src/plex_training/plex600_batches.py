"""Verified, deterministic next-token windows from Plex 600M packed corpora.

This is an input pipeline component, not a trainer. It never creates a model,
optimizer, dataset split, checkpoint or training authorization.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

import torch
from torch import Tensor

from .data import TokenCorpus
from .plex600 import Plex600Config
from .plex600_setup import check_tokenizer_bundle


class Plex600PackedExamples:
    """Read-only record-bounded sampling. The caller selects train or validation.

    Validation is never silently mixed with training. Each sampled pair is
    next-token shifted within one original text record, including its final
    EOS target but never the following record's first token.
    """

    def __init__(self, bundle_dir: Path, *, split: str = "train") -> None:
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.bundle_dir = bundle_dir.resolve(strict=True)
        self.verification = check_tokenizer_bundle(self.bundle_dir)
        self.split = split
        # A manifest/index was SHA/structure-verified before any corpus is opened.
        self.rows = json.loads((self.bundle_dir / f"{split}.index.json").read_text(encoding="utf-8"))
        self.eligible_rows = [row for row in self.rows if row["tokenCount"] >= 2]
        if not self.eligible_rows:
            raise ValueError("No two-token record is available for next-token sampling")
        self.corpus = TokenCorpus(self.bundle_dir / f"{split}.tokens.u16le")
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.corpus.close()
            self._closed = True

    def __enter__(self) -> "Plex600PackedExamples":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample_batch(
        self, rng: random.Random, *, batch_size: int = 1, sequence_length: int = 128,
    ) -> tuple[Tensor, Tensor]:
        if self._closed:
            raise ValueError("Sample source is closed")
        if (type(batch_size) is not int or type(sequence_length) is not int
                or not 1 <= batch_size <= 64
                or not 1 <= sequence_length <= Plex600Config().context_length):
            raise ValueError("Invalid batch or sequence size")
        inputs = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        targets = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
        for index in range(batch_size):
            row = rng.choice(self.eligible_rows)
            # Full-length windows if the record allows, otherwise a right-padded
            # record. Never slide into a second record or shift across EOS.
            valid_length = min(sequence_length, row["tokenCount"] - 1)
            max_start = row["tokenCount"] - valid_length - 1
            offset = rng.randint(0, max_start)
            window = self.corpus._window(row["startToken"] + offset, valid_length + 1)
            inputs[index, :valid_length] = window[:-1]
            targets[index, :valid_length] = window[1:]
        return inputs, targets
