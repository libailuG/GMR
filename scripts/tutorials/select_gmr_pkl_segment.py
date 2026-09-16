#!/usr/bin/env python3
"""在 MuJoCo 中交互式查看并截取 GMR PKL 动作片段。

程序会打开两个窗口：MuJoCo 机器人窗口和时间轴控制窗口。拖动时间轴选择
帧，设置包含首尾帧的选段，预览后保存为新的 PKL 文件。

示例：
    conda run --no-capture-output -n env_gmr_0 python \
        scripts/tutorials/select_gmr_pkl_segment.py \
        outputs/taixi_a2_walk_full.pkl --robot taixi_a2

GMR PKL 中的 root_rot 使用 xyzw；送入 MuJoCo 前会转换成 wxyz，保存时仍保持
原来的 xyzw 格式。
"""

from __future__ import annotations

import argparse
import pickle
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np

from general_motion_retargeting.params import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_file", type=Path, help="bvh_to_robot.py 生成的 GMR PKL")
    parser.add_argument(
        "--robot",
        default="taixi_a2",
        choices=sorted(ROBOT_XML_DICT),
        help="用于回放的机器人模型（默认：taixi_a2）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="可选：覆盖 params.py 中的机器人 MuJoCo XML",
    )
    return parser.parse_args()


def load_gmr_pkl(path: Path) -> dict:
    """加载并检查选段程序需要的 GMR 字段。"""
    if not path.is_file():
        raise FileNotFoundError(f"PKL 文件不存在：{path}")

    with path.open("rb") as file:
        motion = pickle.load(file)
    if not isinstance(motion, dict):
        raise TypeError("PKL 顶层必须是 dict")

    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"PKL 缺少字段：{missing}")

    root_pos = np.asarray(motion["root_pos"])
    root_rot = np.asarray(motion["root_rot"])
    dof_pos = np.asarray(motion["dof_pos"])
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos 形状应为 (帧数, 3)，实际为 {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_rot 形状应为 (帧数, 4)，实际为 {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[0] != root_pos.shape[0]:
        raise ValueError(f"dof_pos 第一维必须等于帧数，实际为 {dof_pos.shape}")
    if root_pos.shape[0] == 0:
        raise ValueError("动作中没有帧")
    if float(motion["fps"]) <= 0:
        raise ValueError(f"fps 必须大于 0，实际为 {motion['fps']}")
    return motion


def crop_motion(motion: dict, start: int, stop: int, frame_count: int) -> dict:
    """裁剪所有第一维等于总帧数的数组；stop 不包含在切片中。"""
    cropped = {}
    for key, value in motion.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == frame_count:
            cropped[key] = value[start:stop].copy()
        elif isinstance(value, list) and len(value) == frame_count:
            cropped[key] = value[start:stop]
        elif isinstance(value, tuple) and len(value) == frame_count:
            cropped[key] = value[start:stop]
        else:
            cropped[key] = value
    return cropped


class GmrPklSegmentSelector:
    def __init__(
        self,
        root: tk.Tk,
        viewer: mj_viewer.Handle,
        model: mj.MjModel,
        data: mj.MjData,
        motion_path: Path,
        motion: dict,
        robot_base: str,
    ) -> None:
        self.root = root
        self.viewer = viewer
        self.model = model
        self.data = data
        self.motion_path = motion_path
        self.motion = motion
        self.robot_base_id = model.body(robot_base).id

        self.root_pos = np.asarray(motion["root_pos"])
        self.root_rot_xyzw = np.asarray(motion["root_rot"])
        self.dof_pos = np.asarray(motion["dof_pos"])
        self.fps = float(motion["fps"])
        self.frame_count = self.root_pos.shape[0]

        expected_dof = model.nq - 7
        if self.dof_pos.shape[1] != expected_dof:
            raise ValueError(
                f"PKL 有 {self.dof_pos.shape[1]} 个关节位置，但模型需要 "
                f"{expected_dof} 个。请检查 --robot/--model 是否正确。"
            )

        self.current_frame = 0
        self.range_start = 0
        self.range_end = self.frame_count - 1  # 包含终点
        self.playing = False
        self.next_frame_time = time.perf_counter()
        self._programmatic_slider = False
        self._closed = False

        self.frame_var = tk.IntVar(value=0)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.follow_camera_var = tk.BooleanVar(value=True)
        self.frame_text = tk.StringVar()
        self.range_text = tk.StringVar()
        self.play_button_text = tk.StringVar(value="播放")

        self._build_ui()
        self._render_frame(0)
        self.root.after(10, self._tick)

    def _build_ui(self) -> None:
        self.root.title(f"GMR PKL 选段 - {self.motion_path.name}")
        self.root.geometry("1000x310")
        self.root.minsize(760, 300)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text=str(self.motion_path)).pack(anchor=tk.W)
        ttk.Label(outer, textvariable=self.frame_text).pack(anchor=tk.W, pady=(8, 0))

        self.timeline = tk.Scale(
            outer,
            from_=0,
            to=self.frame_count - 1,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self._on_slider,
            resolution=1,
            showvalue=False,
            length=920,
        )
        self.timeline.pack(fill=tk.X, pady=(2, 8))
        self.timeline.bind("<ButtonPress-1>", self._on_slider_press)

        playback = ttk.Frame(outer)
        playback.pack(fill=tk.X)
        ttk.Button(playback, textvariable=self.play_button_text, command=self.toggle_play).pack(side=tk.LEFT)
        ttk.Button(playback, text="上一帧", command=lambda: self.step_frame(-1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(playback, text="下一帧", command=lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(playback, text="从选段起点播放", command=self.play_selection).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(playback, text="速度").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Combobox(
            playback,
            textvariable=self.speed_var,
            values=(0.25, 0.5, 1.0, 1.5, 2.0, 4.0),
            width=6,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(playback, text="相机跟随", variable=self.follow_camera_var).pack(side=tk.LEFT, padx=(18, 0))

        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        ttk.Label(outer, textvariable=self.range_text).pack(anchor=tk.W)

        selection = ttk.Frame(outer)
        selection.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(selection, text="当前帧设为起点", command=self.set_start).pack(side=tk.LEFT)
        ttk.Button(selection, text="当前帧设为终点", command=self.set_end).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(selection, text="跳到起点", command=lambda: self.set_frame(self.range_start)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(selection, text="跳到终点", command=lambda: self.set_frame(self.range_end)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(selection, text="重置选段", command=self.reset_range).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(selection, text="保存选段 PKL", command=self.save_selection).pack(side=tk.RIGHT)

        ttk.Label(
            outer,
            text="操作：拖动滑条查看帧；设置起点和终点；预览确认后保存。保存范围包含起点和终点。",
        ).pack(anchor=tk.W, pady=(14, 0))

    def _on_slider_press(self, _event: tk.Event) -> None:
        self.playing = False
        self.play_button_text.set("播放")

    def _on_slider(self, value: str) -> None:
        if self._programmatic_slider or self.playing:
            return
        requested = int(float(value))
        if requested != self.current_frame:
            self.current_frame = requested
            self._render_frame(requested)

    def set_frame(self, frame: int) -> None:
        self.playing = False
        self.play_button_text.set("播放")
        self.current_frame = int(np.clip(frame, 0, self.frame_count - 1))
        self._programmatic_slider = True
        self.timeline.set(self.current_frame)
        self._programmatic_slider = False
        self._render_frame(self.current_frame)

    def step_frame(self, delta: int) -> None:
        self.set_frame(self.current_frame + delta)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button_text.set("暂停" if self.playing else "播放")
        self.next_frame_time = time.perf_counter()

    def play_selection(self) -> None:
        self.set_frame(self.range_start)
        self.playing = True
        self.play_button_text.set("暂停")
        self.next_frame_time = time.perf_counter()

    def set_start(self) -> None:
        self.range_start = self.current_frame
        if self.range_end < self.range_start:
            self.range_end = self.range_start
        self._update_labels()

    def set_end(self) -> None:
        self.range_end = self.current_frame
        if self.range_start > self.range_end:
            self.range_start = self.range_end
        self._update_labels()

    def reset_range(self) -> None:
        self.range_start = 0
        self.range_end = self.frame_count - 1
        self._update_labels()

    def _update_labels(self) -> None:
        current_time = self.current_frame / self.fps
        total_time = (self.frame_count - 1) / self.fps
        self.frame_text.set(
            f"当前帧：{self.current_frame} / {self.frame_count - 1}    "
            f"时间：{current_time:.2f}s / {total_time:.2f}s    FPS：{self.fps:g}"
        )
        selected_frames = self.range_end - self.range_start + 1
        self.range_text.set(
            f"选段：[ {self.range_start}, {self.range_end} ]（包含终点）    "
            f"共 {selected_frames} 帧，时长 {selected_frames / self.fps:.2f}s"
        )

    def _render_frame(self, frame: int) -> None:
        root_quat_xyzw = self.root_rot_xyzw[frame]
        quat_norm = np.linalg.norm(root_quat_xyzw)
        if quat_norm < 1.0e-8:
            raise ValueError(f"第 {frame} 帧根节点四元数为零")

        self.data.qpos[:3] = self.root_pos[frame]
        # GMR PKL: xyzw；MuJoCo qpos: wxyz。
        self.data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]] / quat_norm
        self.data.qpos[7:] = self.dof_pos[frame]
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)

        if self.follow_camera_var.get():
            self.viewer.cam.lookat[:] = self.data.xpos[self.robot_base_id]
        self.viewer.sync()
        self._update_labels()

    def _tick(self) -> None:
        if self._closed:
            return
        if not self.viewer.is_running():
            self.close()
            return

        if self.playing:
            now = time.perf_counter()
            period = 1.0 / (self.fps * self.speed_var.get())
            if now >= self.next_frame_time:
                next_frame = self.current_frame + 1
                if next_frame > self.range_end or next_frame < self.range_start:
                    next_frame = self.range_start
                self.current_frame = next_frame
                self._programmatic_slider = True
                self.timeline.set(self.current_frame)
                self._programmatic_slider = False
                self._render_frame(self.current_frame)
                self.next_frame_time = now + period

        self.root.after(5, self._tick)

    def save_selection(self) -> None:
        start = self.range_start
        stop = self.range_end + 1
        default_name = f"{self.motion_path.stem}_frames_{start}_{self.range_end}.pkl"
        output = filedialog.asksaveasfilename(
            title="保存选段 PKL",
            initialdir=str(self.motion_path.parent),
            initialfile=default_name,
            defaultextension=".pkl",
            filetypes=(("GMR Pickle", "*.pkl"),),
        )
        if not output:
            return

        cropped = crop_motion(self.motion, start, stop, self.frame_count)
        with Path(output).open("wb") as file:
            pickle.dump(cropped, file)
        messagebox.showinfo(
            "保存完成",
            f"已保存 {stop - start} 帧（{(stop - start) / self.fps:.2f}s）：\n{output}",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.playing = False
        try:
            self.viewer.close()
        finally:
            if self.root.winfo_exists():
                self.root.destroy()


def main() -> None:
    args = parse_args()
    motion_path = args.motion_file.expanduser().resolve()
    motion = load_gmr_pkl(motion_path)

    model_path = (
        args.model.expanduser().resolve()
        if args.model is not None
        else Path(ROBOT_XML_DICT[args.robot]).resolve()
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML 不存在：{model_path}")

    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    root = tk.Tk()
    with mj_viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = VIEWER_CAM_DISTANCE_DICT.get(args.robot, 3.0)
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -15.0
        GmrPklSegmentSelector(
            root,
            viewer,
            model,
            data,
            motion_path,
            motion,
            ROBOT_BASE_DICT[args.robot],
        )
        root.mainloop()


if __name__ == "__main__":
    main()
