# Segmentation analysis definitions

These indices describe the **predicted semantic mask**, not its agreement with an expert.
They are separate from the manuscript's image-wise mIoU.

A segment is a connected component of a single class in the final native-resolution
class-ID mask. Default connectivity is 4; `--connectivity 8` changes it explicitly.
No small-component filtering, morphology, smoothing or minimum-size threshold is applied.
Input-derived regions used by A5 are different from the final semantic components.

Segment and class CSVs include all 17 classes, with `included_in_indices` false for
Background (0) and Lines (1). Image-level indices use the remaining architectural
classes. This exclusion applies to all structural image indices in this release;
the manuscript mIoU still includes Background and Lines when present in ground truth.

Let S contain the architectural components, A_s their pixel areas, P_s their
perimeters, and A_D the normalisation area. The default A_D is the full image area
H×W; `--normalisation architectural` uses the total predicted architectural area.
This alternative is prediction-derived, not an independently annotated facade ROI.

| CSV column | Definition |
|---|---|
| N_seg | Number of components in S |
| A_bar_seg | mean(A_s) / A_D |
| CV_area | population SD(A_s) / mean(A_s), ddof=0 |
| L_boundary | sum(P_s), in native pixel-edge units |
| C_boundary | unweighted mean of P_s² / (4π A_s) |
| H_area | −sum(p_c log2 p_c), in bits |
| PP_area | 2^H_area, effective number of area-weighted semantic classes |

Here p_c is the **total pixel area of semantic class c**, summed over all its
components, divided by the total area of classes 2–16. Zero-area classes are
omitted from the entropy sum. Background and Lines are excluded from this
denominator. Splitting a class into more components without changing its total
area does not change H_area or PP_area. Perplexity is not based on component counts.

Perimeters use exposed horizontal/vertical pixel edges, including image borders
and holes. Shared interfaces are counted once for each adjacent segment when
perimeters are summed. This digital perimeter has orientation bias; do not compare
it to a different contour estimator without recalculation. Areas are px² and
perimeters are px, not physical building dimensions. Absolute areas and lengths
depend on drawing scale and raster resolution. A5 inference does not resize inputs.

Centroids are means of pixel-centre coordinates (x+0.5, y+0.5), with the top-left
image corner at (0,0). Bounding boxes are [xmin, ymin, xmax, ymax) with exclusive
maxima. Segment IDs are unique within an image, assigned by ascending class ID then
SciPy connected-component order. `segment_ids/*.npy` maps every pixel to the CSV ID.
The class summary reports all 17 classes and uses population SD (ddof=0).
Absent classes have count=0 and total area=0; their area-distribution statistics
are blank. Images with no architectural pixels have N_seg=L_boundary=0 and
undefined means, CV, complexity, entropy and perplexity (blank CSV / JSON null).

The normalisation scope, connectivity, perimeter convention and source hashes are
recorded in each run's metadata. These implementation choices make the public
generator reproducible; the manuscript did not previously report empirical values
for these structural indices or validate them as human legibility measures.
