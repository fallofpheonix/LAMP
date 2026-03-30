from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from osgeo import ogr, osr

from .terrain import pixel_to_world


@dataclass
class VoxelScene:
    occupancy: np.ndarray  # shape: (nz, rows, cols), 1=solid, 0=air
    z_min: float
    z_res: float
    geotransform: tuple
    projection_wkt: str
    nodata: Optional[float]
    terrain: np.ndarray
    valid_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        nz, rows, cols = self.occupancy.shape
        return nz, rows, cols


def build_voxel_scene(
    terrain: np.ndarray,
    building_heights: np.ndarray,
    geotransform: tuple,
    projection_wkt: str,
    nodata: Optional[float],
    z_res: float = 0.5,
    z_pad_m: float = 2.0,
) -> VoxelScene:
    if terrain.shape != building_heights.shape:
        raise ValueError("terrain and building_heights must be aligned")

    valid = np.isfinite(terrain)
    if nodata is not None:
        valid &= terrain != nodata

    if not np.any(valid):
        raise ValueError("No valid terrain cells")

    terr_valid = terrain[valid]
    bh_valid = np.maximum(building_heights[valid], 0.0)

    z_min = float(np.floor(np.min(terr_valid) - 1.0))
    z_max = float(np.ceil(np.max(terr_valid + bh_valid) + z_pad_m))

    nz = int(np.ceil((z_max - z_min) / z_res)) + 1
    rows, cols = terrain.shape
    occ = np.zeros((nz, rows, cols), dtype=np.uint8)

    # Fill terrain and building solids per column.
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            z_ground = float(terrain[r, c])
            k_ground = int(np.floor((z_ground - z_min) / z_res))
            k_ground = max(0, min(k_ground, nz - 1))
            # Keep the terrain surface voxel as air to allow LOS queries on the surface.
            if k_ground > 0:
                occ[:k_ground, r, c] = 1

            bh = float(max(building_heights[r, c], 0.0))
            if bh > 0.0:
                z_top = z_ground + bh
                k_top = int(np.floor((z_top - z_min) / z_res))
                k_top = max(0, min(k_top, nz - 1))
                # Buildings occupy from terrain surface upward.
                if k_top >= k_ground:
                    occ[k_ground : k_top + 1, r, c] = 1

    return VoxelScene(
        occupancy=occ,
        z_min=z_min,
        z_res=z_res,
        geotransform=geotransform,
        projection_wkt=projection_wkt,
        nodata=nodata,
        terrain=terrain,
        valid_mask=valid,
    )


def carve_openings_from_vector(
    scene: VoxelScene,
    openings_path: str,
    default_radius_m: float = 0.75,
) -> int:
    """
    Carve pass-through openings in occupied voxels from optional vector layer.

    Expected attributes per feature (absolute elevation meters):
      - z_min (required)
      - z_max (required)
      - radius_m (optional)

    Geometry: point preferred (polygon/line uses centroid).
    Returns number of features applied.
    """
    ds = ogr.Open(openings_path)
    if ds is None:
        return 0

    layer = ds.GetLayer(0)
    src_srs = layer.GetSpatialRef()

    dst = osr.SpatialReference()
    dst.ImportFromWkt(scene.projection_wkt)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = None
    if src_srs is not None:
        src = src_srs.Clone()
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not src.IsSame(dst):
            transform = osr.CoordinateTransformation(src, dst)

    origin_x, px_w, _, origin_y, _, px_h = scene.geotransform
    rows, cols = scene.terrain.shape
    nz = scene.occupancy.shape[0]

    used = 0
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        g = geom.Clone()
        if transform is not None:
            g.Transform(transform)

        # Use point directly or centroid for non-point geometry.
        gname = g.GetGeometryName().upper()
        if gname == "POINT":
            x, y, _ = g.GetPoint()
        else:
            ctr = g.Centroid()
            if ctr is None:
                continue
            x, y, _ = ctr.GetPoint()

        z0 = feat.GetField("z_min")
        z1 = feat.GetField("z_max")
        if z0 is None or z1 is None:
            continue

        radius = feat.GetField("radius_m")
        if radius is None:
            radius = default_radius_m

        col_f = (x - origin_x) / px_w
        row_f = (y - origin_y) / px_h
        c0 = int(np.floor(col_f))
        r0 = int(np.floor(row_f))

        rad_cells = max(0, int(np.ceil(float(radius) / abs(px_w))))
        k0 = int(np.floor((float(min(z0, z1)) - scene.z_min) / scene.z_res))
        k1 = int(np.floor((float(max(z0, z1)) - scene.z_min) / scene.z_res))
        k0 = max(0, min(k0, nz - 1))
        k1 = max(0, min(k1, nz - 1))

        rr0 = max(0, r0 - rad_cells)
        rr1 = min(rows - 1, r0 + rad_cells)
        cc0 = max(0, c0 - rad_cells)
        cc1 = min(cols - 1, c0 + rad_cells)

        if rr0 <= rr1 and cc0 <= cc1 and k0 <= k1:
            scene.occupancy[k0 : k1 + 1, rr0 : rr1 + 1, cc0 : cc1 + 1] = 0
            used += 1

    return used


def _world_to_indices(scene: VoxelScene, x: float, y: float, z: float) -> tuple[int, int, int]:
    origin_x, px_w, _, origin_y, _, px_h = scene.geotransform
    col = int(np.floor((x - origin_x) / px_w))
    row = int(np.floor((y - origin_y) / px_h))
    k = int(np.floor((z - scene.z_min) / scene.z_res))
    return k, row, col


def _inside(scene: VoxelScene, k: int, row: int, col: int) -> bool:
    nz, rows, cols = scene.occupancy.shape
    return 0 <= k < nz and 0 <= row < rows and 0 <= col < cols


def is_visible_3d(
    scene: VoxelScene,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    step_m: Optional[float] = None,
) -> bool:
    x0, y0, z0 = start_xyz
    x1, y1, z1 = end_xyz

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    length = float(np.sqrt(dx * dx + dy * dy + dz * dz))

    if length <= 1e-9:
        return True

    if step_m is None:
        step_m = min(abs(scene.geotransform[1]), scene.z_res)
    step_m = max(step_m, 0.2)

    n_steps = int(np.ceil(length / step_m))
    if n_steps <= 1:
        return True

    for i in range(1, n_steps):
        t = i / n_steps
        x = x0 + dx * t
        y = y0 + dy * t
        z = z0 + dz * t

        k, row, col = _world_to_indices(scene, x, y, z)
        if not _inside(scene, k, row, col):
            return False
        if scene.occupancy[k, row, col] != 0:
            return False

    return True


def compute_ground_viewshed_3d(
    scene: VoxelScene,
    obs_row: int,
    obs_col: int,
    observer_height: float,
    target_height: float,
    max_distance_m: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    rows, cols = scene.terrain.shape
    if not (0 <= obs_row < rows and 0 <= obs_col < cols and scene.valid_mask[obs_row, obs_col]):
        raise ValueError("Observer is outside valid terrain")

    x_obs, y_obs = pixel_to_world(scene.geotransform, obs_row, obs_col)
    z_obs = float(scene.terrain[obs_row, obs_col] + max(observer_height, 0.05))

    out = np.zeros((rows, cols), dtype=np.uint8)

    checked = 0
    visible = 0
    px = abs(scene.geotransform[1])

    for r in range(rows):
        for c in range(cols):
            if not scene.valid_mask[r, c]:
                continue
            if max_distance_m is not None:
                if np.hypot(r - obs_row, c - obs_col) * px > max_distance_m:
                    continue

            x_t, y_t = pixel_to_world(scene.geotransform, r, c)
            z_t = float(scene.terrain[r, c] + max(target_height, 0.05))
            checked += 1

            if is_visible_3d(scene, (x_obs, y_obs, z_obs), (x_t, y_t, z_t)):
                out[r, c] = 1
                visible += 1

    stats = {
        "checked_cells": checked,
        "visible_cells": visible,
        "visible_ratio": (visible / checked) if checked else 0.0,
        "observer_elevation_m": z_obs,
    }
    return out, stats


def compute_visibility_volume_from_observer(
    scene: VoxelScene,
    obs_row: int,
    obs_col: int,
    observer_height: float,
    azimuth_steps: int = 360,
    elevation_steps: int = 91,
    elev_min_deg: float = -30.0,
    elev_max_deg: float = 60.0,
    max_distance_m: Optional[float] = None,
) -> np.ndarray:
    """
    Ray-march air voxels visible from observer; returns uint8 volume (1=visible air).
    """
    rows, cols = scene.terrain.shape
    if not (0 <= obs_row < rows and 0 <= obs_col < cols and scene.valid_mask[obs_row, obs_col]):
        raise ValueError("Observer is outside valid terrain")

    x_obs, y_obs = pixel_to_world(scene.geotransform, obs_row, obs_col)
    z_obs = float(scene.terrain[obs_row, obs_col] + max(observer_height, 0.05))

    nz, _, _ = scene.shape
    visible = np.zeros_like(scene.occupancy, dtype=np.uint8)

    if max_distance_m is None:
        # Scene diagonal with margin.
        max_distance_m = float(np.hypot(rows * abs(scene.geotransform[1]), cols * abs(scene.geotransform[1])) + 10.0)

    step = max(min(abs(scene.geotransform[1]), scene.z_res), 0.5)
    n_steps = int(np.ceil(max_distance_m / step))

    az = np.linspace(0.0, 2.0 * np.pi, azimuth_steps, endpoint=False)
    el = np.deg2rad(np.linspace(elev_min_deg, elev_max_deg, elevation_steps))

    for a in az:
        ca = np.cos(a)
        sa = np.sin(a)
        for e in el:
            ce = np.cos(e)
            se = np.sin(e)
            dx = ce * ca
            dy = ce * sa
            dz = se

            for i in range(1, n_steps + 1):
                t = i * step
                x = x_obs + dx * t
                y = y_obs + dy * t
                z = z_obs + dz * t

                k, r, c = _world_to_indices(scene, x, y, z)
                if not _inside(scene, k, r, c):
                    break

                if scene.occupancy[k, r, c] != 0:
                    break

                visible[k, r, c] = 1

    # Ensure observer air voxel is marked visible.
    k_obs, r_obs, c_obs = _world_to_indices(scene, x_obs, y_obs, z_obs)
    if _inside(scene, k_obs, r_obs, c_obs) and scene.occupancy[k_obs, r_obs, c_obs] == 0:
        visible[k_obs, r_obs, c_obs] = 1

    # Only air voxels are meaningful.
    visible[scene.occupancy != 0] = 0
    return visible
