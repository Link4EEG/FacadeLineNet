# Licences and attribution

Copyright (c) 2026 Ju Hyun Lee, Michael J. Ostwald and Seung Yeul Ji.
Author affiliations and sources are listed in [AUTHORS.md](AUTHORS.md).

## Scope

- **MIT** ([full licence](../LICENSE)): project-specific software, scripts, tests,
  build configuration and documentation, except the assets listed below.
- **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  ([full licence](../LICENSE-CC-BY-4.0.txt)): A5 model checkpoints distributed in
  `weights/` or as release attachments; numerical results in `paper_results/`;
  metadata and palettes; schematic example image and generated example outputs.

These licences apply to different components, not as a choice of either licence
for the entire package. The archive contains both components. Attribute the
research authors, cite the software DOI, retain applicable licence notices and
indicate modifications as required by the relevant licence. Licence texts govern.

The original commercial-source architectural drawings and expert masks are not
included. These licences do not grant rights over those excluded source materials.
Third-party dependencies retain their own licences, including PyTorch,
torchvision, NumPy, SciPy and Pillow. The complete A5 checkpoint contains a model
whose ResNet-34 encoder was initialised from torchvision's ImageNet weights before
task-specific training. No institutional endorsement is implied by affiliation.
