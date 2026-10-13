# Archive a fixed release on Zenodo

GitHub is the working repository. Zenodo preserves a particular software version
and its A5 checkpoints and assigns a DOI upon publication. No DOI has been issued
for this package yet.

For the first release, a manual software upload is convenient because the archive
includes the code and all three binary model files. A GitHub source-code archive
does not contain the separately attached checkpoint release assets; verify actual
Zenodo file contents rather than assuming the model files were included.

1. Sign in to https://zenodo.org/ using GitHub, ORCID or a Zenodo account. Complete
   email verification if requested.
2. Create a new upload. Upload `FacadeLineNet-v1.0.0-complete.zip` and
   `SHA256SUMS.txt` from the GitHub v1.0.0 release (or the prepared local assets).
3. Set resource type to **Software**, version **1.0.0**, and title to
   **FacadeLineNet A5 facade segmentation and structural index generator**.
4. Enter creators in manuscript order: Ju Hyun Lee, Michael J. Ostwald,
   Ji Seung Yeul. Confirm each author's affiliation and any ORCID identifiers.
   Participating institutions are UNSW Sydney and Hanyang University.
5. Select the author-approved licence. The prepared source metadata should agree
   with the licence of the actual archive; do not invent a licence or DOI.
6. Set the GitHub release URL as a related identifier and reserve a DOI with
   **Get a DOI now**. A reserved DOI is not yet a published record.
7. Save the draft, check the files, authors, licence, version and description,
   then publish. Add the issued DOI to README and CITATION.cff in the working
   GitHub repository. Keep the tagged version unchanged and use a new version
   for subsequent file updates.

For later automatic source-code archiving, connect GitHub under the Zenodo account
settings, allow the Link4EEG organization if needed, enable this repository, then
create a new GitHub release. Keep the code archive and binary checkpoint archive
linked and verify that the intended model files are present in the archived record.
Do not create duplicate DOI records for the same release.

Official guidance:
- https://help.zenodo.org/docs/deposit/create-new-upload/
- https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://help.zenodo.org/docs/github/describe-software/zenodo-json/
