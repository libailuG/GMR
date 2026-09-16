"""
A2 Height Scanner Demo — with real-time MuJoCo viewer.

Loads the A2 model, drops it onto the ground, and scans terrain
heights beneath the robot using the 187-ray height scanner.

Controls:
  Space   toggle height scan overlay
  R       reset robot pose
  Esc     close viewer

Usage:
    python demo_height_scanner.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from height_scanner import HeightScanner

# ── 1. Load the A2 model ─────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("a2.xml")
data = mujoco.MjData(model)

scanner = HeightScanner(model, data)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

# ── 2. Initial robot pose ────────────────────────────────────────────────
def reset_pose():
    data.qpos[:] = 0.0
    data.qpos[2] = 1.0     # z
    data.qpos[3] = 1.0     # qw
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

reset_pose()

# ── 3. Viewer setup ──────────────────────────────────────────────────────
show_overlay = True
reset_flag = False
_lock = __import__("threading").Lock()

def key_callback(key: int):
    """Handle keyboard input (runs in viewer thread)."""
    global show_overlay, reset_flag
    if key == ord(" "):
        show_overlay = not show_overlay
    elif key in (ord("R"), ord("r")):
        with _lock:
            reset_flag = True

viewer = mujoco.viewer.launch_passive(
    model, data,
    key_callback=key_callback,
    show_left_ui=False,
    show_right_ui=False,
)

# Camera
viewer.cam.azimuth = 90
viewer.cam.elevation = -20
viewer.cam.distance = 4.0
viewer.cam.lookat[:] = [0.0, 0.0, 0.6]

# ── 4. Main loop ─────────────────────────────────────────────────────────
print("=" * 60)
print("  A2 Height Scanner Demo")
print("=" * 60)
print(f"  Rays: {scanner.num_rays}  |  Joints: {model.njnt}  |  Actuators: {model.nu}")
print("  [Space] toggle overlay   [R] reset   [Esc] quit")
print()

step_count = 0
t0 = time.perf_counter()

# Font / gridpos constants
FONT_NORMAL = mujoco.mjtFontScale.mjFONTSCALE_100
GRID_TOPLEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT

while viewer.is_running():
    # ── Handle reset (deferred from key callback) ─────────────────
    with _lock:
        do_reset = reset_flag
        reset_flag = False
    if do_reset:
        reset_pose()
        step_count = 0
        t0 = time.perf_counter()

    # ── Step physics ─────────────────────────────────────────────
    mujoco.mj_step(model, data)
    step_count += 1

    # ── Build overlay text every 10 steps ────────────────────────
    if step_count % 10 == 0 and show_overlay:
        base_z = data.xpos[base_id][2]
        heights = scanner.scan(max_distance=30.0, base_height=base_z)

        elapsed = time.perf_counter() - t0
        fps = step_count / elapsed if elapsed > 0 else 0

        lines = [
            f"time:  {elapsed:.1f}s   fps: {fps:.0f}",
            f"base_z: {base_z:.3f} m",
            f"h_min:  {heights.min():+.3f} m",
            f"h_max:  {heights.max():+.3f} m",
            f"h_mean: {heights.mean():+.3f} m",
            f"h_std:  {heights.std():.4f} m",
        ]

        viewer.set_texts([
            (FONT_NORMAL, GRID_TOPLEFT, "A2 Height Scanner", ""),
            *[(FONT_NORMAL, GRID_TOPLEFT, line, "") for line in lines],
        ])
    elif not show_overlay:
        viewer.set_texts([])

    # ── Sync viewer ──────────────────────────────────────────────
    viewer.sync()

viewer.close()
print("\n  ✓ Demo complete.")
