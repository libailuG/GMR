"""在 MuJoCo 中直观查看 GMR 人体动作字典的坐标。

颜色约定：X 红、Y 绿、Z 蓝。

示例：
    conda activate env_gmr_0
    python scripts/tutorials/visualize_human_coordinates.py
    python scripts/tutorials/visualize_human_coordinates.py --frame 100 --body RightHand
    python scripts/tutorials/visualize_human_coordinates.py --play

默认显示全部关节坐标系；如只想看 World、Hips 和选中部位：
    python scripts/tutorials/visualize_human_coordinates.py --no-all-axes

键盘：
    空格      暂停/播放
    左/右方向键  前一帧/后一帧
    R         回到命令行指定的起始帧
"""

import argparse
import time

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting.robot_motion_viewer import draw_frame
from general_motion_retargeting.utils.lafan1 import load_bvh_file


# LAFAN1 骨架的父子连接，只用于绘制黄色骨架线。
SKELETON_EDGES = [
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToe"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToe"),
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("Spine2", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
]


def add_sphere(viewer, position, radius, color, label=None):
    """在 user scene 中添加关节点小球。"""
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=position,
        mat=np.eye(3).flatten(),
        rgba=np.asarray(color, dtype=float),
    )
    if label is not None:
        geom.label = label
    viewer.user_scn.ngeom += 1


def add_line(viewer, start, end, width=0.012, color=(1.0, 0.8, 0.1, 1.0)):
    """在两个关节点之间添加骨架连线。"""
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([width, width, width]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(color, dtype=float),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=start,
        to=end,
    )
    viewer.user_scn.ngeom += 1


def draw_human(viewer, frame, selected_body, axis_size, all_axes):
    """绘制骨架、关节点、世界坐标轴和身体局部坐标轴。"""
    viewer.user_scn.ngeom = 0

    for parent, child in SKELETON_EDGES:
        if parent in frame and child in frame:
            add_line(viewer, frame[parent][0], frame[child][0])

    for body_name, (position, _) in frame.items():
        if body_name.endswith("Mod"):
            continue
        is_selected = body_name in {"Hips", selected_body}
        add_sphere(
            viewer,
            position,
            radius=0.025 if is_selected else 0.015,
            color=(1.0, 0.25, 0.1, 1.0) if is_selected else (0.85, 0.85, 0.85, 1.0),
            label=body_name if is_selected else None,
        )

    # 世界坐标轴方向处处相同。为了让它靠近人物，这里把它画在
    # Hips 的地面投影点，而不是遥远的数学原点 (0, 0, 0)。
    hips_position = frame["Hips"][0]
    world_axes_position = np.array([hips_position[0], hips_position[1], 0.0])
    draw_frame(
        world_axes_position,
        np.eye(3),
        viewer,
        axis_size * 1.25,
        joint_name="World XYZ (at ground)",
    )

    axes_to_draw = list(frame) if all_axes else ["Hips", selected_body]
    for body_name in dict.fromkeys(axes_to_draw):
        if body_name.endswith("Mod"):
            continue
        position, quaternion = frame[body_name]
        rotation_matrix = Rotation.from_quat(
            quaternion, scalar_first=True
        ).as_matrix()
        draw_frame(
            position,
            rotation_matrix,
            viewer,
            axis_size * 0.45 if all_axes and body_name not in {"Hips", selected_body} else axis_size,
            joint_name=body_name,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="在 MuJoCo 中查看人体位置、旋转和坐标轴")
    parser.add_argument("--bvh-file", default="lafan1/walk1_subject1.bvh")
    parser.add_argument("--frame", type=int, default=0, help="起始帧，从 0 开始")
    parser.add_argument("--body", default="LeftHand", help="重点显示坐标轴的身体部位")
    parser.add_argument("--play", action="store_true", help="启动后立即播放；默认暂停")
    parser.add_argument(
        "--all-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示所有关节坐标轴（默认开启；使用 --no-all-axes 关闭）",
    )
    parser.add_argument("--axis-size", type=float, default=0.22, help="坐标轴长度，单位为米")
    parser.add_argument("--fps", type=float, default=30.0, help="播放帧率")
    return parser.parse_args()


def main():
    args = parse_args()
    frames, _ = load_bvh_file(args.bvh_file, format="lafan1")

    if not 0 <= args.frame < len(frames):
        raise ValueError(f"--frame 应在 0 到 {len(frames) - 1} 之间")
    if args.body not in frames[0]:
        raise ValueError(f"未知身体部位 {args.body!r}，可用名称：{list(frames[0])}")

    # 创建一个只有地面和灯光的简单 MuJoCo 场景，避免机器人遮挡人体坐标。
    scene_xml = """
    <mujoco model="human_coordinate_viewer">
      <visual><global azimuth="135" elevation="-15"/></visual>
      <worldbody>
        <light pos="0 0 5" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="20 20 0.1"
              rgba="0.18 0.20 0.24 1"/>
      </worldbody>
    </mujoco>
    """
    model = mj.MjModel.from_xml_string(scene_xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    state = {
        "frame": args.frame,
        "start_frame": args.frame,
        "playing": args.play,
    }

    def on_key(keycode):
        if keycode == glfw.KEY_SPACE:
            state["playing"] = not state["playing"]
        elif keycode == glfw.KEY_RIGHT:
            state["playing"] = False
            state["frame"] = min(state["frame"] + 1, len(frames) - 1)
        elif keycode == glfw.KEY_LEFT:
            state["playing"] = False
            state["frame"] = max(state["frame"] - 1, 0)
        elif keycode in (glfw.KEY_R, ord("r")):
            state["playing"] = False
            state["frame"] = state["start_frame"]

    print("颜色：X 红、Y 绿、Z 蓝")
    print("操作：空格=暂停/播放，左右方向键=逐帧，R=回到起始帧")
    print(f"重点观察：{args.body}；起始帧：{args.frame}")

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=on_key,
    ) as viewer:
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        while viewer.is_running():
            loop_start = time.perf_counter()
            frame = frames[state["frame"]]
            viewer.cam.lookat = frame["Hips"][0]

            draw_human(viewer, frame, args.body, args.axis_size, args.all_axes)
            viewer.sync()

            if state["playing"]:
                state["frame"] = (state["frame"] + 1) % len(frames)

            remaining = 1.0 / args.fps - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
