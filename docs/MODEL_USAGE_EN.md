# FacadeLineNet A5 (Seed 0): A Concise User Guide

This package provides the trained A5 seed 0 checkpoint and the minimal code required for image pre-processing and semantic segmentation inference. It can be used without downloading additional project files from GitHub. Quantitative analysis tools are not included. The suffix `v2` denotes a revision to the distribution package; it does not indicate a newly trained model.

## 1. Installation

Extract the ZIP archive and open a terminal in the directory containing `segment.py`. Python 3.10 or later is required; the reference environment uses Python 3.12.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `source .venv/bin/activate`.

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

For NVIDIA GPU inference, first install compatible PyTorch and torchvision builds for the available hardware. CPU inference is also supported, although processing large drawings may take substantially longer.

## 2. Segmenting a Drawing

Prepare an architectural facade line drawing with a light background as a raster image, such as PNG, JPEG, TIFF or BMP. Convert PDF or CAD drawings to raster images before processing. The model was developed for line drawings rather than conventional photographs of buildings. Expert-annotated masks and the seed 1 and seed 2 checkpoints are not required for inference.

```bash
python segment.py --input "drawing.png" --output "results/drawing" --device cpu
```

Omit `--device cpu` to enable automatic selection of an available CUDA GPU. Adjust the input and output paths as required. The output directory must not already exist; this restriction protects previously generated results. By default, the script uses the supplied checkpoint, `weights/A5_seed0.pt`.

## 3. Interpreting the Outputs

| File | Description |
|---|---|
| `class_ids.png` | Class ID mask at the original image dimensions, with pixel values from 0 to 16 |
| `segmentation.png` | Colour-coded segmentation mask using the 17-class palette |
| `overlay.png` | Segmentation mask superimposed on the input drawing |
| `prediction.json` | Checkpoint, device, numerical precision and image dimensions used for inference |

Class names, definitions and colours are provided in `metadata/palette.json`. As the class ID values are small, `class_ids.png` may appear dark in a standard image viewer. Use `segmentation.png` or `overlay.png` for visual inspection.

## 4. Processing Method and Scope

The pre-processing pipeline derives greyscale intensity, distance-to-line and region-area channels from the input drawing. Inference uses 512-pixel tiles and contextual windows, assembles predictions at the original image resolution, and applies region-consistency processing. The pipeline does not impose a reduction in the output image dimensions.

Large drawings may require several gigabytes of system RAM. If GPU memory is insufficient, use CPU inference or review the input image dimensions. Any rescaling changes the input resolution and may affect predictions. Automated segmentation results require expert review.

The package includes the seed 0 checkpoint, model architecture, pre-processing and inference code, colour mapping and an execution script. It excludes quantitative analysis and structural-index calculations, training code, other seed checkpoints, and the original research drawings and expert-annotated masks.

Model weights and palettes are distributed under CC BY 4.0. Code and documentation are distributed under the MIT licence. Consult the accompanying licence files for the applicable terms and attribution requirements.

## 5. Updating the Zenodo Files

The revised archive includes pre-processing and inference code and therefore has a different checksum from the earlier package. Replace both `FacadeLineNet-models_v2.zip` and `FacadeLineNet-models_v2.sha256` on Zenodo with the corresponding revised files. This guide is included in the ZIP archive; uploading it separately is optional.
