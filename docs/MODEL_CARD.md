# FacadeLineNet A5 model card

FacadeLineNet predicts 17 semantic classes from facade line drawings at their
native raster resolution. The architecture combines a ResNet-34 semantic encoder,
ASPP, a U-Net-style decoder, a high-resolution detail branch, gated fusion and a
learned region-consistency layer. Model state keys and numerical architecture are
preserved from the manuscript implementation.

Inputs are grayscale drawing intensity, log distance to ink and log input-cell
area, normalised to [-1,1]. Non-ink regions use 4-connectivity. Ink is gray<128,
falling back to gray<200 when the ink fraction is below 0.0005. Distance saturates
at 64 native pixels. Derived channels are quantised to uint8 as in the original.
The detail tile is 512×512; the co-centred context window is 2048×2048 reduced by
4×4 area averaging. Tiles use stride 384 and aligned last positions. Interior
margins are discarded, border extents retained, and probabilities assembled in
float16. Region consistency operates on whole-image cells on CPU. Input ink is
assigned class 1 (Lines). No expert mask is needed at inference.

Checkpoints: seed 0 iteration 6000, seed 1 iteration 4000, seed 2 iteration 7000.
Seed 0 is the default; these are independent runs, not an ensemble. The original
experiment used PyTorch 2.7.0, torchvision 0.22.0 and an NVIDIA RTX 3090. CUDA
auto precision prefers bf16 where supported; CPU uses float32. Different hardware
or precision can cause small score differences. Training initially used ImageNet
encoder weights; full checkpoints are supplied, so inference never downloads an
ImageNet checkpoint.

The corpus has 395 paired drawings, with fixed train/validation/test counts
276/58/61. Source collections included commercial architectural assets. Original
drawings and expert masks are not bundled here. The bundled schematic is newly
drawn solely to demonstrate the interface, and is not a held-out research sample.

Native image-wise mean mIoU: U-Net 0.769263; Hybrid v2 0.857019; A5 seeds 0/1/2
0.880184 / 0.874323 / 0.887411. Three-seed mean ± sample SD is 0.880639 ± 0.006556.
The metric excludes ignored GT pixels and averages IoU over GT-present classes
within each image, then over images. It includes Background and Lines. Predicted
classes absent in GT are not separately added as zero-valued terms.

These are internal development results on a repeatedly consulted test set, not an
untouched external benchmark. Baseline capacity and pretraining differ. Rare
architectural classes remain unreliable, particularly Detailing and Overhangs.
Structural indices inherit prediction errors. Outputs require architectural review;
no independent human validation or demonstrated annotation time saving is claimed.
