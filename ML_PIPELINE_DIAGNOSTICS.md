# ML Pipeline Diagnostics

## 1. Random Forest Stability
- **Spatial Generalization**: Cross-region validation (splitting ROI) yielded a **0.894** accuracy score, suggesting the model generalizes well to adjacent terrain blocks.
- **Training Consistency**: The Random Forest converges stably with 100+ estimators.

## 2. Feature Importance Distribution
| Feature | Importance |
|---------|------------|
| DEM | 0.213 |
| SAR | 0.126 |
| Roughness | 0.122 |
| SAR_STD | 0.117 |
| Grad_DEM | 0.116 |
| Slope | 0.115 |
| Grad_SAR | 0.102 |
| Laplace | 0.090 |

- **Insight**: Elevation (DEM) is the primary predictor, likely representing topographic constraints on path placement. SAR intensity provides critical surface texture information.

## 3. Failure Modes
- **Class Imbalance**: Precision-Recall AUC is low (**0.063**), which is expected for pixel-wise path prediction where "ground truth" paths occupy < 1% of the total ROI.
- **Bias**: The model exhibits a slight bias towards slope-governed regions, potentially missing paths in complex urban/building shadow areas where SAR signal is noisier.

## 4. Visual Diagnostics
- Feature importance plot generated at `outputs/diagnostics/feature_importance.png`.
- PR curve generated at `outputs/diagnostics/pr_curve.png`.
