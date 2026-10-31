# Plex 600M — data readiness review

**Status: read-only review; no 600M corpus or tokenizer approved or built.**
This reviews recorded source metadata and existing preparation code for the
separate scratch 600M research track. Plex Nano's 27,566,080-parameter model
remains the controlled baseline. For coding evaluations, the caller supplies
the relevant HTML, CSS or JavaScript context and request.

## Historical corpus that could be considered

[P2-26 closeout](PHASE-2-P2-26-CLOSEOUT.md) records a 27M-era Web corpus with
3,140 records (3,071 train and 69 validation), approximately 5.21 MiB of
accepted normalized source text, six repositories and three source groups.
The four MDN repositories share one group; the other two groups are Bulma
Templates and JavaScript30. The exact pinned revisions and historical license
evidence are in the [source batch record](PHASE-2-P2-26-SOURCE-BATCH-2.md).

On October 31, 2026, the local dataset **manifest** was present and its SHA-256
matched the closeout: `2f02f199d101050c9939df6d389851e46bbd858157cb68d7cb8cc75863178a91`.
This check did not rehash either JSONL split, revalidate notices or repeat the
historical contamination check. The original P2-26 report documents its own
filtered build and zero matches under its then-declared protected-source rules;
it did not open the final holdout and does not establish absence of all
overlap with future material.

The corpus is a possible historical input for review, not a representative
large-scale 600M dataset. Three source groups and one validation group limit
the diversity of evidence. Its validation split was already measured during
[P2-27 tokenizer selection](PHASE-2-P2-27-RESULT.md), so it is not an untouched
benchmark. Past owner rights reviews and corpus promotion have their recorded
scope; they are not a new 600M corpus decision. No quantitative claim about the
data size required for training this model follows from this review.

## Existing preparation path and gaps

The [600M setup CLI](../training/src/plex_training/plex600_setup.py) uses the
shared tokenizer builder, which accepts the historical manifest's `p1-14.2`
format, checks declared split hashes and counts, verifies packaged license
notice hashes and approved source metadata, checks group and exact-content
separation, and fits BPE only on training text. These checks validate declared
inputs; they do not decide whether source rights or content are suitable for
the new purpose. The 32,768 vocabulary is a capacity, and no full 600M
tokenizer has been fitted.

The 600M bundle builder now records a SHA-256 for each packed-record index.
The checker verifies the same index bytes it parses, rejects duplicate record
IDs, and the sampler uses the verified index snapshot. Older 600M bundles that
lack those hashes are refused; they are not silently modified. This binds an
index to its bundle manifest, not to an independently authenticated manifest
or verified source-record mapping. A jointly changed manifest and index can
still pass. The subsequent [input-integrity work](PLEX-600M-INPUT-INTEGRITY-PLAN.md)
replaces live token mappings with bounded verified snapshots and adds an explicit
source-record check using independently supplied manifest hashes. This is not
an automatic source or rights certification when constructing a sampler.

Before relying on a dataset for 600M work, review and record:

1. The intended use, sources, exact revisions, license notices, rights review,
   and what content each source actually permits and contributes.
2. A source-family-based split chosen before fitting, with related material
   grouped together. Define an independent development evaluation before
   exposure; label historical, already-inspected validation as such. Keep the
   final project holdout closed.
3. A reproducible manifest, content hashes, source counts and byte/token mix;
   review duplicate, secret, content-quality and protected-evaluation checks
   with their coverage limits. Exact deduplication is not near-duplicate proof.
4. The tokenizer's source and resulting bundle identities, train-only fit,
   roundtrips and context/encoding behavior. A bundle integrity check is not
   independent evidence that the corpus is suitable or that the model can code.
5. A separately reviewed run contract and distinct authorization before any
   optimizer update or training execution. A one-pass GPU check does not
   establish optimizer-state or sustained-training fit.

These are review criteria, not an instruction to fetch sources, generate data,
fit a tokenizer, run inference, or train a model. No such work occurred during
this review. The prior Nano experimental results and artifacts remain unchanged.
