"""
Height Scanner for MuJoCo A2 model.

Replicates Isaac Lab's RayCasterCfg height scanner using mujoco.mj_ray().
Configured to match:
  RayCasterCfg(
      prim_path="{ENV_REGEX_NS}/Robot/base_link",
      offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
      ray_alignment="yaw",
      pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
  )

Usage:
    from height_scanner import HeightScanner

    scanner = HeightScanner(model, data)
    heights = scanner.scan()  # returns (187,) array of heights relative to base_link
"""

from __future__ import annotations

import numpy as np
from collections.abc import Sequence


class HeightScanner:
    """Height scanner using MuJoCo ray casting.

    Casts rays downward from a grid of sites attached to the robot's base_link
    and returns the distance from each ray origin to the first geom hit.

    Ray alignment: "yaw" — rays always point straight down in the WORLD frame
    (i.e., gravity direction), regardless of robot orientation.
    """

    # Grid configuration (must match MJCF site positions)
    GRID_SIZE_X = 1.6       # meters
    GRID_SIZE_Y = 1.0       # meters
    GRID_RESOLUTION = 0.1   # meters
    Z_OFFSET = 20.0         # ray origin height above base_link
    RAY_DIRECTION = np.array([0.0, 0.0, -1.0])  # world-frame: straight down

    def __init__(self, model, data):
        """
        Args:
            model: mujoco.MjModel
            data: mujoco.MjData
        """
        import mujoco

        self._model = model
        self._data = data

        # Resolve height scanner site IDs
        self._site_ids: list[int] = []
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name and name.startswith("site_hs_"):
                self._site_ids.append(i)

        self._n_rays = len(self._site_ids)
        if self._n_rays == 0:
            raise RuntimeError(
                "No height scanner sites found in model. "
                "Ensure the MJCF contains site_hs_* sites."
            )
        # Expected: 17 × 11 = 187
        expected = ((int(self.GRID_SIZE_X / self.GRID_RESOLUTION) + 1) *
                     (int(self.GRID_SIZE_Y / self.GRID_RESOLUTION) + 1))
        if self._n_rays != expected:
            import warnings
            warnings.warn(
                f"HeightScanner: found {self._n_rays} sites, expected {expected} "
                f"({int(self.GRID_SIZE_X / self.GRID_RESOLUTION) + 1}×"
                f"{int(self.GRID_SIZE_Y / self.GRID_RESOLUTION) + 1} grid)"
            )

        # Pre-allocate output buffers
        self._heights = np.zeros(self._n_rays, dtype=np.float64)
        # Ray direction in [3, 1] shape (mj_ray requirement)
        self._ray_dir_3x1 = self.RAY_DIRECTION.copy().reshape(3, 1)
        # Pre-allocate geomid output buffer (mj_ray writes hit geom id here)
        self._geomid_buf = np.zeros(1, dtype=np.int32).reshape(1, 1)

    @property
    def num_rays(self) -> int:
        """Number of rays in the grid."""
        return self._n_rays

    def scan(self, max_distance: float = 25.0, base_height: float = 0.0) -> np.ndarray:
        """Cast all rays and return heights relative to base_link.

        Each ray is cast from its site's world position straight down.
        The returned height is:
            ray_origin_z - hit_point_z - base_height
        which gives the terrain height *below* the robot's base.

        Rays that miss all geoms (no contact) return -max_distance.

        Args:
            max_distance: Maximum ray length (m). Rays that miss return this.
            base_height: Base link reference height to subtract.

        Returns:
            (num_rays,) float64 array of terrain heights below base.
        """
        import mujoco

        m = self._model
        d = self._data

        # Ray direction in world frame (straight down)
        ray_dir = self.RAY_DIRECTION

        for i, site_id in enumerate(self._site_ids):
            # Ray origin and direction in [3, 1] shape required by mj_ray
            origin = d.site_xpos[site_id].copy().reshape(3, 1)
            ray_vec = self._ray_dir_3x1

            # Cast ray against the scene
            # mj_ray returns distance to nearest geom, or -1 on miss
            dist = mujoco.mj_ray(
                m, d,
                origin,         # ray origin (world, [3, 1])
                ray_vec,        # ray direction (world, [3, 1])
                None,                # geomgroup: None = all groups
                1,                   # flg_static: 1 = include static geoms
                -1,                  # bodyexclude: -1 = none excluded
                self._geomid_buf,    # geomid: output buffer [1, 1]
            )

            if dist < 0:
                # Ray missed — use max_distance to indicate "far below"
                self._heights[i] = -max_distance
            else:
                # Hit point: origin + dist * direction
                hit_z = origin[2, 0] + dist * ray_vec[2, 0]
                # Height = how far below the base the terrain is
                self._heights[i] = hit_z - base_height

        return self._heights

    def scan_relative(self, max_distance: float = 25.0) -> np.ndarray:
        """Cast rays and return distance from each site origin to the hit point.

        Positive values mean terrain is BELOW the ray origin (typical).
        Rays that miss return max_distance.

        Args:
            max_distance: Distance to return for rays that miss.

        Returns:
            (num_rays,) float64 array of distances along ray direction.
        """
        import mujoco

        m = self._model
        d = self._data

        for i, site_id in enumerate(self._site_ids):
            origin = d.site_xpos[site_id].copy().reshape(3, 1)
            dist = mujoco.mj_ray(m, d, origin, self._ray_dir_3x1, None, 1, -1, self._geomid_buf)
            self._heights[i] = dist if dist >= 0 else max_distance

        return self._heights

    def get_ray_origins(self) -> np.ndarray:
        """Get current world-frame positions of all ray origin sites.

        Returns:
            (num_rays, 3) float64 array.
        """
        origins = np.zeros((self._n_rays, 3), dtype=np.float64)
        for i, site_id in enumerate(self._site_ids):
            origins[i] = self._data.site_xpos[site_id]
        return origins


# ---------------------------------------------------------------------------
#  Convenience function: direct usage without class
# ---------------------------------------------------------------------------

def compute_height_scan(
    model,
    data,
    grid_size: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
    z_offset: float = 20.0,
    base_body_name: str = "base_link",
    max_distance: float = 25.0,
) -> np.ndarray:
    """Standalone function: compute height scan using mujoco.mj_ray().

    This is a reference implementation for sim-to-sim transfer.
    It dynamically computes ray origins instead of using pre-defined sites.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        grid_size: (x_size, y_size) in meters.
        resolution: Grid spacing in meters.
        z_offset: Height of ray origins above base_link (m).
        base_body_name: Name of the base body.
        max_distance: Max ray length for rays that miss.

    Returns:
        (num_rays,) float64 array of distances along rays.
    """
    import mujoco

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_id < 0:
        raise ValueError(f"Body '{base_body_name}' not found in model.")

    # Build grid in base_link frame
    nx = int(grid_size[0] / resolution) + 1
    ny = int(grid_size[1] / resolution) + 1
    xs = np.linspace(-grid_size[0] / 2, grid_size[0] / 2, nx)
    ys = np.linspace(-grid_size[1] / 2, grid_size[1] / 2, ny)

    # Base pose
    base_pos = data.xpos[base_id]
    base_quat = data.xquat[base_id]  # w x y z — MuJoCo convention

    # Rotate local offsets to world frame
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])  # x y z w

    ray_dir = np.array([0.0, 0.0, -1.0])
    heights = np.zeros(nx * ny, dtype=np.float64)

    idx = 0
    for y in ys:
        for x in xs:
            local_origin = np.array([x, y, z_offset])
            world_origin = base_pos + rot.apply(local_origin)

            # Cast ray
            dist = mujoco.mj_ray(
                model, data,
                world_origin.reshape(3, 1),  # [3, 1] required
                ray_dir.reshape(3, 1),        # [3, 1] required
                None, 1, -1,                   # geomgroup, flg_static, bodyexclude
                np.zeros(1, dtype=np.int32).reshape(1, 1),  # geomid buffer
            )
            heights[idx] = dist if dist >= 0 else max_distance
            idx += 1

    return heights
