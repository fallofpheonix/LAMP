# Limitations and Roadmap

## Current Limitations
1. **Voxel 3D, not mesh 3D**
- 3D visibility currently uses voxel occupancy, not explicit triangle meshes/BVH.
- Architectural fine geometry (thin walls, arches, overhang edges) is approximated.

2. **No explicit opening transmissivity**
- Optional opening carving is supported only via a manual openings vector layer.
- Project-provided openings are not yet fully encoded as structured input.

3. **Baseline learned surrogate only**
- Logistic model is implemented and trained, but it is a low-capacity baseline.
- Higher-capacity models (GBDT/CNN/GNN) are not yet integrated.

4. **No audibility modeling**
- Out of scope for this repository version.

## Upgrade Path
1. **True 3D geometry**
- Build triangle-mesh scene (terrain + wall solids + apertures).
- Add BVH-accelerated ray engine.

2. **Opening-aware visibility**
- Add explicit opening layer derived from site plan annotation.
- Apply transmissivity by face material class.

3. **ML surrogate**
- Generate deterministic training corpus from ray engine.
- Train fast predictor for `P(visible | observer, cell)`.

4. **Formal benchmark suite**
- Add synthetic canonical tests and error thresholds.
