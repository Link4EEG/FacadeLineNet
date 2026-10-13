# Reproducibility

The model and module implementations preserve the A5 state-dictionary layout.
Preprocessing and tiled assembly were extracted from the research implementation,
with a standalone Drawing adapter replacing machine-specific cache paths. CPU
execution disables CUDA autocast. Inference loads checkpoints with
`torch.load(..., weights_only=True)` and strict state-key validation.

Release validation on the development machine included:
- Analytical component/perimeter/centroid/bounding-box tests.
- Semantic-area entropy/perplexity tests, including unequal class areas.
- Connectivity, holes, no-architecture masks and area-normalisation tests.
- Native GT-present mIoU tests, including ignored pixels and a hallucinated class.
- Recursive batch analysis and output identity checks.
- A5 checkpoint loading for seeds 0, 1 and 2.
- Single-image and batch inference on an original schematic drawing.
- Comparison with one native research drawing: input channels matched the original
  cached channels exactly, including context windows. Seed-0 mIoU was
  0.8521294895280389 versus archived 0.8521377515711092 (difference −0.0000082620),
  on a 2861×2216 image with CUDA bf16.

This is a targeted port-validation check, not a new evaluation of all 61 images.
Original aggregate accuracy and full per-image scores remain in `paper_results/`.
The archival CSV was computed from native U-Net/Hybrid masks and original A5
evaluation records; `aggregate_results.csv` documents zero differences from the
stored means. Rounded per-class IoUs were not used to reconstruct image mIoU.

`metadata/weights_manifest.json` pins each checkpoint's SHA-256 and original byte
size. Train/validation/test filenames are provided without redistributing the
underlying source drawings. Architecture and inference are included; this release
does not claim to include an end-to-end retraining dataset or pipeline.

The public structural-index definitions are fixed in
`docs/SEGMENTATION_ANALYSIS_INDICES.md`. They implement the requested repository
framework. They do not reproduce undocumented formulas from the separate SAM
wrapper, which required an unavailable base module.
