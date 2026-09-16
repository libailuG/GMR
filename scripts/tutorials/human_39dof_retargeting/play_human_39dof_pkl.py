#!/usr/bin/env python3
"""播放独立重映射程序生成的 human_39dof GMR PKL。"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import glfw
import numpy as np

from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROBOT_KEY = "human_39dof_standalone_player"
MODEL_PATH = REPO_ROOT / "assets" / "human_39dof" / "mujoco" / "human.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_file", type=Path, help="human_39dof GMR PKL")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--frame", type=int, default=0, help="起始帧")
    parser.add_argument("--paused", action="store_true", help="启动后保持暂停")
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="相机是否跟随骨盆",
    )
    return parser.parse_args()


def load_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"PKL 不存在：{path}")
    with path.open("rb") as file:
        motion = pickle.load(file)
    if not isinstance(motion, dict):
        raise TypeError("PKL 顶层必须是 dict")

    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"PKL 缺少字段：{missing}")

    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=float)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=float)
    dof_pos = np.asarray(motion["dof_pos"], dtype=float)
    frame_count = root_pos.shape[0]

    if fps <= 0:
        raise ValueError(f"fps 必须大于 0，实际为 {fps}")
    if root_pos.shape != (frame_count, 3):
        raise ValueError(f"root_pos 应为 (帧数, 3)，实际为 {root_pos.shape}")
    if root_rot_xyzw.shape != (frame_count, 4):
        raise ValueError(f"root_rot 应为 (帧数, 4)，实际为 {root_rot_xyzw.shape}")
    if dof_pos.shape != (frame_count, 39):
        raise ValueError(f"dof_pos 应为 (帧数, 39)，实际为 {dof_pos.shape}")
    if frame_count == 0:
        raise ValueError("PKL 中没有动作帧")
    if not all(np.isfinite(array).all() for array in (root_pos, root_rot_xyzw, dof_pos)):
        raise ValueError("PKL 中存在 NaN 或 Inf")

    # GMR PKL 保存为 xyzw；RobotMotionViewer/MuJoCo 使用 wxyz。
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    root_rot_wxyz /= np.linalg.norm(root_rot_wxyz, axis=1, keepdims=True)
    return fps, root_pos, root_rot_wxyz, dof_pos


def register_standalone_model() -> None:
    """只在当前进程中临时注册，不修改 params.py。"""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")
    ROBOT_XML_DICT[ROBOT_KEY] = MODEL_PATH
    ROBOT_BASE_DICT[ROBOT_KEY] = "PelvisLink"
    VIEWER_CAM_DISTANCE_DICT[ROBOT_KEY] = 3.0


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed 必须大于 0")

    fps, root_pos, root_rot, dof_pos = load_motion(args.motion_file)
    frame_count = root_pos.shape[0]
    if not 0 <= args.frame < frame_count:
        raise ValueError(f"--frame 应在 0 到 {frame_count - 1} 之间")

    register_standalone_model()
    state = {
        "frame": args.frame,
        "start_frame": args.frame,
        "playing": not args.paused,
    }

    def on_key(keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            state["playing"] = not state["playing"]
        elif keycode == glfw.KEY_RIGHT:
            state["playing"] = False
            state["frame"] = min(state["frame"] + 1, frame_count - 1)
        elif keycode == glfw.KEY_LEFT:
            state["playing"] = False
            state["frame"] = max(state["frame"] - 1, 0)
        elif keycode in (glfw.KEY_R, ord("r")):
            state["playing"] = False
            state["frame"] = state["start_frame"]

    viewer = RobotMotionViewer(
        robot_type=ROBOT_KEY,
        motion_fps=fps * args.speed,
        transparent_robot=0,
        keyboard_callback=on_key,
    )
    print(f"文件：{args.motion_file.resolve()}")
    print(f"帧数：{frame_count}，FPS：{fps:g}，速度：{args.speed:g}x")
    print("操作：空格=播放/暂停，左右方向键=逐帧，R=回到起始帧")

    try:
        while viewer.viewer.is_running():
            loop_start = time.perf_counter()
            frame = state["frame"]
            viewer.step(
                root_pos=root_pos[frame],
                root_rot=root_rot[frame],
                dof_pos=dof_pos[frame],
                rate_limit=False,
                follow_camera=args.follow_camera,
            )

            if state["playing"]:
                state["frame"] = (frame + 1) % frame_count

            remaining = 1.0 / (fps * args.speed) - (
                time.perf_counter() - loop_start
            )
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
