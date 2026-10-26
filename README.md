# FacadeLineNet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23329504.svg)](https://doi.org/10.5281/zenodo.23329504)

FacadeLineNet A5 performs native-resolution semantic segmentation of architectural
façade line drawings into 17 classes. The current [Zenodo model package](https://doi.org/10.5281/zenodo.23329504)
contains **seed 0 only**, together with minimal pre-processing and single-image
inference code. Seed 1 and seed 2 are not required to use this model.

This GitHub repository also retains the broader research software, including
batch processing, quantitative analysis and supplementary results. Those functions
are separate from the current model-focused Zenodo download.

## Start with the model package

Download `FacadeLineNet-models_v2.zip` and `FacadeLineNet-models_v2.sha256`
from [Zenodo](https://zenodo.org/records/23329504). Extract the ZIP, open its
`FacadeLineNet-models_v2` folder, and run:

```bash
python -m pip install -r requirements.txt
python segment.py --input "drawing.png" --output "results/drawing" --device cpu
```

The output directory must not already exist. Omit `--device cpu` to select an
available CUDA GPU automatically. Outputs are `class_ids.png`, `segmentation.png`,
`overlay.png` and `prediction.json`. No additional project download is required.
See the [concise user guide](docs/MODEL_USAGE_EN.md). The commands above apply to
the extracted Zenodo package; the sections below describe the broader GitHub software.

Research authors (manuscript order):

- **Ju Hyun Lee** (1).
- **Michael J. Ostwald** (1).
- **Seung Yeul Ji** (1, 2).

1. School of Built Environment, Faculty of Arts, Design & Architecture, UNSW Sydney, Sydney, Australia.
2. School of Architecture, Hanyang University, Seoul, Republic of Korea.

See [author affiliations and official sources](docs/AUTHORS.md).
Software and documentation: **MIT**. A5 weights, numerical research results,
metadata and schematic example assets: **CC BY 4.0**.
See [licence scope and attribution](docs/RIGHTS.md).

## Install

Python 3.10 or later is required. Python 3.12, PyTorch 2.7.0 and torchvision 0.22.0
are the reference environment. NVIDIA CUDA is recommended for large drawings;
CPU inference is supported but slower. Install a matching PyTorch/torchvision pair
for your hardware using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Link4EEG/FacadeLineNet.git
cd FacadeLineNet
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/download_weights.py --seed 0
```

All three standalone A5 checkpoints remain available in the historical
[v1.0.0 release](https://github.com/Link4EEG/FacadeLineNet/releases/tag/v1.0.0),
approximately 115 MB each. The downloader uses those verified attachments.
Use `--seed 1` or `--seed 2` to download another run. SHA-256 checksums are verified
automatically. An archived bundle includes `weights/` directly and needs no download.

## One image

```bash
python scripts/segment_single.py --input examples/schematic_facade.png --output results/single --weights weights/A5_seed0.pt
```

## A folder of images

```bash
python scripts/batch_segment.py --input input_facades --output results/batch --weights weights/A5_seed0.pt
```

PNG, JPEG, TIFF and BMP inputs are discovered recursively. Transparent images are
composited onto white. Inputs retain their raster dimensions. Output must be outside
the input directory. Use a fresh output directory, or explicitly add `--overwrite`.
Any image error stops the command with a nonzero exit code rather than silently
omitting a drawing. Input-relative paths and extensions are preserved in output
names (for example `house.jpg.png`) to avoid collisions.

`--device cpu` forces CPU inference. `--precision fp32` disables GPU autocast.
`--connectivity 4` and `--normalisation image` are the index defaults.
Full-image probability assembly uses host memory proportional to 17×H×W;
large drawings can require several GB of RAM.

## Outputs

| Path | Content |
|---|---|
| segmentations/ | 17-class RGB semantic masks at native dimensions |
| class_ids/ | uint8 PNG masks, class IDs 0–16 |
| overlays/ | Input RGB blended with semantic colours (alpha 0.45) |
| segment_ids/ | uint32 NumPy arrays linking pixels to segment CSV IDs |
| facade_segments.csv | Class, segment ID, area, perimeter, centroid, bounding box, complexity |
| facade_class_summary.csv | 17 rows per image: count, total area, min/max/mean/median/SD of component area |
| facade_segmentation_analysis_results.csv | N_seg, A_bar_seg, CV_area, L_boundary, C_boundary, H_area, PP_area |
| run_metadata.json | Version, input/checkpoint hashes, dimensions and calculation conventions |

Image-level structural indices exclude Background and Lines. Entropy and
perplexity use **total semantic-class areas**, not segment counts. See
[exact formulas and coordinate conventions](docs/SEGMENTATION_ANALYSIS_INDICES.md).

Existing class-ID masks can be analyzed without loading a neural model:

```bash
python scripts/generate_segmentation_analysis.py --input results/single/class_ids --output results/analysis_only
```

## Classes and evaluation

[Palette and definitions](metadata/palette.json) specify all 17 classes and RGB
colours. [Model card](docs/MODEL_CARD.md) documents inference and limitations.
`paper_results/test_image_results.xlsx` and `.csv` are Supplementary Data S1:
61 test images with U-Net, Hybrid v2 and A5 seed 0/1/2 native image-wise mIoU.
These archived accuracy results are separate from the structural indices above.

For locally held expert masks, evaluate native class-ID predictions with:

```bash
python scripts/evaluate_test_set.py --predictions results/batch/class_ids --ground-truth expert_masks --split metadata/test_split.txt --output results/evaluation.csv
python -m unittest discover -s tests -v
```

Original research images and expert masks are not distributed. The original
schematic example is illustrative; it is not evidence of model accuracy.

## Releases and citation

The current Zenodo deposit at **https://doi.org/10.5281/zenodo.23329504**
provides the seed 0 model and minimal segmentation inference tools as a single
`FacadeLineNet-models_v2.zip`, its `.sha256` file and an English user guide.
The `v2` suffix identifies a packaging revision; the checkpoint remains model
version 1.0.1 and its parameters are unchanged. No split-volume restoration is
required. Quantitative-analysis tools, Supplementary Data S1 and the other seeds
are not included in this Zenodo package.

The [GitHub v1.0.1 release](https://github.com/Link4EEG/FacadeLineNet/releases/tag/v1.0.1)
now has the same three uploaded files as Zenodo: the model ZIP, its checksum
and the English user guide. Download the model ZIP rather than GitHub's
automatically generated source-code archives, which do not include model weights.

For the broader software archive, three standalone checkpoints and supplementary
results, use the historical [v1.0.0 release](https://github.com/Link4EEG/FacadeLineNet/releases/tag/v1.0.0).
The model parameters are unchanged. Earlier complete archives contain historical
author metadata; use the corrected affiliations in the current model package and
[author documentation](docs/AUTHORS.md). See [package details](docs/ZENODO_RELEASE.md).

Suggested citation for the current Zenodo model deposit:
Lee, J. H., Ostwald, M. J., & Ji, S. Y. (2026). *FacadeLineNet: Model for Semantic
Segmentation of Architectural Façade Line Drawings* (Version 1.0.1) [Model and
inference code]. Zenodo. https://doi.org/10.5281/zenodo.23329504

This software archive accompanies a manuscript under peer review. Software DOI
registration does not imply acceptance or peer-review certification of that manuscript.
