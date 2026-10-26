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

## GitHub release files and historical materials

The [GitHub v1.0.1 release](https://github.com/Link4EEG/FacadeLineNet/releases/tag/v1.0.1)
has the same three uploaded files as Zenodo: the model ZIP, its checksum and the
English user guide. GitHub also displays automatically generated source-code
archives; these do not include weights and are not the model download.

The historical [GitHub v1.0.0 release](https://github.com/Link4EEG/FacadeLineNet/releases/tag/v1.0.0)
retains the complete software archive, all three standalone A5 checkpoints,
supplementary results and examples. The standalone checkpoint parameters are
identical to those previously attached to v1.0.1. The v1.0.0 complete ZIP has
SHA-256 `f7d6b7844f1347e48f0d44df79148f5cd00bd2e58f00bb6846f431cfd088c3eb`;
it is a different historical package from the removed v1.0.1 complete ZIP.
Use the corrected author affiliations in current documentation rather than
historical embedded author metadata.

The former v1.0.1 complete ZIP, three standalone checkpoints and historical
`SHA256SUMS.txt` have been removed from the v1.0.1 release attachments.
Local research copies are retained. Git tags are unchanged.
The current model ZIP is self-contained and needs no other checkpoint download.
The downloader in the current GitHub branch uses verified v1.0.0 attachments.

Zenodo previously distributed the complete v1.0.1 ZIP as 31 split volumes with
restoration scripts and checksums. That layout has been replaced by the current
three-file model deposit. No split restoration is now required.

Use the current record title when citing the model deposit. Future changes to
trained parameters should be versioned separately from packaging-only changes.
Archival publication does not certify acceptance or peer review of the manuscript.
