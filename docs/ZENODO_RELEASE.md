# Fixed archive on Zenodo

GitHub hosts current code; the v1.0.1 release packages a fixed snapshot and all
three A5 checkpoints. The first Zenodo archive is published:
https://doi.org/10.5281/zenodo.23329504 (version 1.0.1).
The all-versions DOI is https://doi.org/10.5281/zenodo.23329503.
Use the version DOI when citing this exact research software release.

The complete ZIP includes source code, licences, author metadata, numerical
results, schematic examples and model weights. A GitHub-generated source ZIP
alone excludes the separately attached model files.

For reliable transfer, Zenodo hosts the complete ZIP as 31 binary split volumes
(`.zip.001` through `.zip.031`) plus a restoration script and checksum manifests.
Download all volumes, `restore_archive.py` and `split_manifest.json` into one
folder and run `python restore_archive.py`, or open `.zip.001` with 7-Zip.
The rebuilt ZIP is byte-identical to the complete GitHub v1.0.1 release asset.
All 36 files on Zenodo were verified against local MD5 checksums before publication;
local restoration was additionally verified against the original ZIP's SHA-256.

The archived files retain their pre-publication wording about DOI reservation.
This page and the current README/CITATION.cff record the completed publication;
the v1.0.1 tag and archived release files remain unchanged.

The archive is component-licensed: software/documentation MIT; model weights,
numerical data, metadata and schematic assets CC BY 4.0. Both licences are
recorded on Zenodo; see RIGHTS.md for exact scope. Author sources are in AUTHORS.md.

For subsequent changes create a new version, retain the existing tagged snapshot,
and update the version DOI after publication. Do not create a duplicate record
for this v1.0.1 archive. GitHub automatic integration is optional; verify that any
future archive actually contains the model files.

Official guidance:
- https://help.zenodo.org/docs/deposit/create-new-upload/
- https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
