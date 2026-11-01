# Plex 600M: M7 owner-reported first learning run

**Date:** 2026-10-31. **Evidence provenance:** values pasted by repository owner from local PowerShell. The report.json and weights.pt have not been independently received here; checksum values are not invented.

The single explicitly authorized M7 run completed four real-data optimizer updates (protocol plex600-m7-four-update-v1). The source manifest was pinned to SHA-256 4bb5742df59f0c56aaac34fc3129012b4617acf7d86a1804e47c5aa69e541a07 and the saved tokenizer bundle manifest SHA-256 to 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55. Original owner-authored data only; all six questionable external sources, old P2 validation, and the final holdout stayed excluded.

| Update | Pre-update loss | Supervised positions |
| ---: | ---: | ---: |
| 1 | 10.591265678405762 | 55 |
| 2 | 9.79226303100586 | 41 |
| 3 | 8.087221145629883 | 70 |
| 4 | 7.351552486419678 | 54 |
| **Total** | **different training records, not a controlled loss curve** | **220** |

Internal validation on 24 records and 1,744 target positions improved from 10.59880861667318 to 8.972932472688342, a reduction of 1.6258761439848382 (15.340178342565244%). Post-training selected-window train loss 6.830017410625111. Update-phase throughput 210.32593252614265 supervised positions/second; peak CUDA-reserved bytes 11,253,317,632 (10.48046875 GiB).

The owner reported a successful local checksum comparison for the checkpoint. **No original checkpoint SHA-256, report SHA-256, checkpoint size or independently verified original artifact** is present in this conversation. The [owner-reported data receipt](../training/pretraining/plex600-m7-owner-reported-result-2026-10-31.json) deliberately keeps unknown hashes null.

The newly added read-only local verifier can stream-hash actual weights and validate report structure without training:

~~~powershell
git pull --ff-only origin master
uv run --project training --no-sync python -m plex_training.plex600_research m7-receipt --report training/artifacts/plex600-m7-first-four-update/report.json --checkpoint training/artifacts/plex600-m7-first-four-update/weights.pt
~~~

Do not call the checkpoint an exact optimizer resume: it includes model weights only, not AdamW state. The four pre-update losses come from different records, so their decrease is not a same-input training curve. The fixed internal validation decrease is evidence of changed token predictions, **not proof of useful independent coding edits**. No additional training has been authorized.

See [M8 research readiness](PLEX-600M-M8-RESEARCH-READINESS.md) for the separate development evaluation, quarantined training candidates and next gates.
