# Model-focused Zenodo deposit

The current deposit is https://doi.org/10.5281/zenodo.23329504 (model version 1.0.1).
The all-versions DOI is https://doi.org/10.5281/zenodo.23329503.

## Download files

- `FacadeLineNet-models_v2.zip` (approximately 106.9 MB).
- `FacadeLineNet-models_v2.sha256`: SHA-256 checksum of that ZIP.
- `FacadeLineNet-models_v2_USAGE_EN.md`: concise guide in Australian English.

The ZIP includes the A5 seed 0 checkpoint, model structure, pre-processing,
native-resolution tiled inference, region-consistency processing, colour mapping,
`segment.py`, dependencies, class definitions, documentation and licence texts.
The English guide is also included in the ZIP as `USAGE_EN_AU.md`.
See [usage instructions](MODEL_USAGE_EN.md).

This package excludes quantitative-analysis tools, structural-index computation,
batch-processing tools, Supplementary Data S1, other seed checkpoints and examples.
Original research drawings and expert masks are not distributed.

## Model version and packaging revision

Only the distribution scope has changed. `v2` denotes the packaging revision;
the model remains version 1.0.1. Seed 0 is unchanged and can be used independently.
Model weights, metadata and palettes use CC BY 4.0; code and documentation use MIT.
The corrected author affiliations are provided in the package README and the
current record; see [author information](AUTHORS.md).

## Historical complete release

The original [GitHub v1.0.1 release](https://github.com/Link4EEG/FacadeLineNet/releases/tag/v1.0.1)
retains the complete software archive, all three checkpoints, supplementary
results and examples. Those tagged files and release attachments remain unchanged.
Its full-archive SHA-256 is
`326eab1a80e0700a33974efab2da74d0810e2e76ef5e1180b5a77945d765260d`.

Zenodo previously distributed that complete ZIP as 31 split volumes with
restoration scripts and checksums. Those files have been replaced by the current
three-file model deposit. No split restoration is now required, and the current
Zenodo ZIP is not byte-identical to the historical complete GitHub ZIP.

Use the current record title when citing the model deposit. Future changes to
trained parameters should be versioned separately from packaging-only changes.
Archival publication does not certify acceptance or peer review of the manuscript.
