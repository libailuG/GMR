"""在 MuJoCo 中同时显示人体目标与 Unitree G1，直观理解 IK 映射。

颜色与形状：
    黄色骨架/橙色球：缩放、坐标校正后的人体 IK 目标
    青色球：G1 对应 Body/Link 的实际位置
    紫色连线：人体目标与机器人 Link 的对应关系及位置误差
    RGB 箭头：X 红、Y 绿、Z 蓝；长轴是人体目标，短轴是机器人实际朝向

运行：
    conda activate env_gmr_0
    python scripts/tutorials/visualize_human_g1_mapping.py
    python scripts/tutorials/visualize_human_g1_mapping.py --human-body LeftFootMod
    python scripts/tutorials/visualize_human_g1_mapping.py --play

键盘：空格暂停/播放，左右方向键逐帧，R 回到起始帧。
"""

import argparse
import time

import glfw
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
from general_motion_retargeting.robot_motion_viewer import draw_frame
from general_motion_retargeting.utils.lafan1 import load_bvh_file


# 缩放后的 GMR 目标只保留参与 IK 的关键部位，因此使用简化骨架连接。
TARGET_SKELETON_EDGES = [
    ("Hips", "Spine2"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFootMod"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFootMod"),
    ("Spine2", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
]


def add_sphere(viewer, position, radius, color, label=None):
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


def add_line(viewer, start, end, width, color):
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


def mapping_pairs(retargeter):
    """返回 (人体部位, 机器人 Body/Link) 对。"""
    return [
        (entry[0], robot_body)
        for robot_body, entry in retargeter.ik_match_table2.items()
    ]


def robot_body_pose(viewer, robot_body):
    body_id = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_BODY, robot_body)
    return (
        viewer.data.xpos[body_id].copy(),
        viewer.data.xmat[body_id].reshape(3, 3).copy(),
    )


def draw_mapping_scene(
    viewer,
    human_targets,
    pairs,
    selected_human_body,
    axis_size,
    all_axes,
):
    viewer.viewer.user_scn.ngeom = 0
    scene = viewer.viewer

    # 1. 人体目标骨架。
    for parent, child in TARGET_SKELETON_EDGES:
        add_line(
            scene,
            human_targets[parent][0],
            human_targets[child][0],
            width=0.011,
            color=(1.0, 0.72, 0.05, 0.9),
        )

    # 2. 每个人体目标与对应 G1 Link 的位置、坐标轴及误差连线。
    for human_body, robot_body in pairs:
        target_position, target_quaternion = human_targets[human_body]
        robot_position, robot_matrix = robot_body_pose(viewer, robot_body)
        selected = human_body == selected_human_body

        add_sphere(
            scene,
            target_position,
            radius=0.036 if selected else 0.022,
            color=(1.0, 0.25, 0.05, 1.0),
            label=f"Human target: {human_body}" if selected else None,
        )
        add_sphere(
            scene,
            robot_position,
            radius=0.032 if selected else 0.018,
            color=(0.0, 0.9, 1.0, 1.0),
            label=f"G1 link: {robot_body}" if selected else None,
        )
        add_line(
            scene,
            target_position,
            robot_position,
            width=0.012 if selected else 0.004,
            color=(1.0, 0.0, 0.9, 1.0) if selected else (0.65, 0.2, 1.0, 0.55),
        )

        if all_axes or selected or human_body == "Hips":
            target_matrix = Rotation.from_quat(
                target_quaternion, scalar_first=True
            ).as_matrix()
            # 长坐标轴：人体校正后的目标朝向。
            draw_frame(
                target_position,
                target_matrix,
                scene,
                axis_size if selected else axis_size * 0.55,
                joint_name=f"H:{human_body}" if selected else None,
            )
            # 短坐标轴：机器人 Link 的实际朝向。
            draw_frame(
                robot_position,
                robot_matrix,
                scene,
                axis_size * 0.72 if selected else axis_size * 0.38,
                joint_name=f"R:{robot_body}" if selected else None,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="同时显示人体 IK 目标和 Unitree G1")
    parser.add_argument("--bvh-file", default="lafan1/walk1_subject1.bvh")
    parser.add_argument("--frame", type=int, default=0, help="起始帧，从 0 开始")
    parser.add_argument("--human-body", default="LeftHand", help="重点高亮的人体部位")
    parser.add_argument("--play", action="store_true", help="启动后立即播放")
    parser.add_argument(
        "--all-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示全部映射坐标轴（默认开启；使用 --no-all-axes 关闭）",
    )
    parser.add_argument("--axis-size", type=float, default=0.22, help="重点坐标轴长度/米")
    parser.add_argument("--fps", type=float, default=30.0, help="播放帧率")
    parser.add_argument(
        "--opaque-robot",
        action="store_true",
        help="使用不透明机器人；默认半透明，便于观察内部目标",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames, human_height = load_bvh_file(args.bvh_file, format="lafan1")
    if not 0 <= args.frame < len(frames):
        raise ValueError(f"--frame 应在 0 到 {len(frames) - 1} 之间")

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        verbose=False,
    )
    pairs = mapping_pairs(retargeter)
    available_human_bodies = [human_body for human_body, _ in pairs]
    if args.human_body not in available_human_bodies:
        raise ValueError(
            f"{args.human_body!r} 没有映射到 G1。可用名称：{available_human_bodies}"
        )

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

    print("黄色骨架/橙色球=人体 IK 目标，青色球=G1 Link")
    print("紫线=映射及位置误差；RGB 长轴=人体目标，RGB 短轴=机器人实际朝向")
    print("操作：空格=暂停/播放，左右方向键=逐帧，R=回到起始帧")
    selected_robot_body = dict(pairs)[args.human_body]
    print(f"重点映射：{args.human_body} -> {selected_robot_body}")

    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        motion_fps=args.fps,
        transparent_robot=0 if args.opaque_robot else 1,
        keyboard_callback=on_key,
    )

    last_solved_frame = None
    solved_qpos = None
    human_targets = None
    try:
        while viewer.viewer.is_running():
            loop_start = time.perf_counter()
            frame_index = state["frame"]

            # 暂停时不重复求解同一帧，保证画面稳定。
            if frame_index != last_solved_frame:
                solved_qpos = retargeter.retarget(frames[frame_index])
                human_targets = retargeter.scaled_human_data
                last_solved_frame = frame_index

            viewer.data.qpos[:] = solved_qpos
            mj.mj_forward(viewer.model, viewer.data)
            pelvis_id = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_BODY, "pelvis")
            viewer.viewer.cam.lookat = viewer.data.xpos[pelvis_id]
            viewer.viewer.cam.distance = 2.2
            viewer.viewer.cam.elevation = -10

            draw_mapping_scene(
                viewer,
                human_targets,
                pairs,
                args.human_body,
                args.axis_size,
                args.all_axes,
            )
            viewer.viewer.sync()

            if state["playing"]:
                state["frame"] = (state["frame"] + 1) % len(frames)

            remaining = 1.0 / args.fps - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
