"""逐帧观察 LAFAN1 人体动作到 Taixi A2 的动作重映射。

这个程序默认暂停。每按一次左右方向键，只切换并求解一帧，适合检查：

1. 人体部位是否映射到了正确的 A2 Body/Link；
2. 人体目标位置和 A2 实际位置之间还有多少误差；
3. 人体目标坐标轴和 A2 Link 坐标轴是否对齐；
4. 左右脚底是否穿过地面；
5. 12 个腿部关节是否到达限位。

画面说明：
    黄色骨架、橙色球：经过缩放和偏置后的人体 IK 目标；
    青色球：A2 对应 Link 的实际位置；
    紫色线：人体目标与 A2 Link 的映射线（也是位置误差）；
    RGB 长轴：人体目标朝向，X 红、Y 绿、Z 蓝；
    RGB 短轴：A2 Link 的实际朝向；
    绿色球：A2 左右脚底接触点。

运行：
    conda activate env_gmr_0
    cd /home/libai/09_gmr/GMR
    python scripts/tutorials/visualize_human_taixi_a2_mapping.py

观察其他映射：
    python scripts/tutorials/visualize_human_taixi_a2_mapping.py \
        --human-body LeftFootMod

使用固定世界视角（相机不跟随机器人）：
    python scripts/tutorials/visualize_human_taixi_a2_mapping.py \
        --no-follow-camera

键盘：
    右方向键：下一帧
    左方向键：上一帧
    空格：播放或暂停
    R：回到命令行指定的起始帧
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


# GMR 处理后只保留参与 IK 的下半身关键点，因此使用这棵简化骨架。
# 每个元组表示一条仅供显示的“父部位 -> 子部位”连线。
LOWER_BODY_EDGES = [
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFootMod"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFootMod"),
]


def add_sphere(scene, position, radius, color, label=None):
    """向 MuJoCo user scene 添加一个不参与物理计算的标记球。"""
    geom = scene.user_scn.geoms[scene.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=np.asarray(position),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(color, dtype=float),
    )
    if label is not None:
        geom.label = label
    scene.user_scn.ngeom += 1


def add_line(scene, start, end, width, color):
    """用胶囊几何体画一条三维线段。"""
    geom = scene.user_scn.geoms[scene.user_scn.ngeom]
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
        from_=np.asarray(start),
        to=np.asarray(end),
    )
    scene.user_scn.ngeom += 1


def get_mapping_pairs(retargeter):
    """把 JSON 中的映射转换为 (人体部位, A2 Body/Link) 列表。"""
    return [
        (entry[0], robot_body)
        for robot_body, entry in retargeter.ik_match_table2.items()
    ]


def get_robot_body_pose(viewer, body_name):
    """读取 A2 Body/Link 的世界位置、旋转矩阵和 wxyz 四元数。"""
    body_id = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_BODY, body_name)
    return (
        viewer.data.xpos[body_id].copy(),
        viewer.data.xmat[body_id].reshape(3, 3).copy(),
        viewer.data.xquat[body_id].copy(),
    )


def draw_current_frame(
    viewer,
    human_targets,
    pairs,
    selected_human_body,
    axis_size,
    all_axes,
):
    """在 A2 模型上叠加当前帧的人体目标和映射信息。"""
    scene = viewer.viewer
    # 每帧先清空自定义几何体，否则球、线和坐标轴会不断累积。
    scene.user_scn.ngeom = 0

    # 黄色线表示人体下半身目标骨架，不参与 IK。
    for parent, child in LOWER_BODY_EDGES:
        add_line(
            scene,
            human_targets[parent][0],
            human_targets[child][0],
            width=0.011,
            color=(1.0, 0.72, 0.05, 0.9),
        )

    for human_body, robot_body in pairs:
        target_position, target_quaternion = human_targets[human_body]
        robot_position, robot_matrix, _ = get_robot_body_pose(viewer, robot_body)
        selected = human_body == selected_human_body

        # 橙色球：真正送给 IK 的人体目标位置。
        add_sphere(
            scene,
            target_position,
            radius=0.040 if selected else 0.024,
            color=(1.0, 0.25, 0.05, 1.0),
            label=f"Human target: {human_body}" if selected else None,
        )
        # 青色球：用 qpos 正运动学得到的 A2 Link 实际位置。
        add_sphere(
            scene,
            robot_position,
            radius=0.034 if selected else 0.019,
            color=(0.0, 0.9, 1.0, 1.0),
            label=f"A2 link: {robot_body}" if selected else None,
        )
        # 紫线越长，说明这个映射的位置误差越大。
        add_line(
            scene,
            target_position,
            robot_position,
            width=0.013 if selected else 0.004,
            color=(1.0, 0.0, 0.9, 1.0) if selected else (0.65, 0.2, 1.0, 0.55),
        )

        if all_axes or selected or human_body == "Hips":
            # GMR 四元数是 wxyz；SciPy 必须设置 scalar_first=True。
            target_matrix = Rotation.from_quat(
                target_quaternion, scalar_first=True
            ).as_matrix()
            # 长 RGB 轴表示目标朝向。
            draw_frame(
                target_position,
                target_matrix,
                scene,
                axis_size if selected else axis_size * 0.58,
                joint_name=f"H:{human_body}" if selected else None,
            )
            # 短 RGB 轴表示 A2 实际朝向。
            draw_frame(
                robot_position,
                robot_matrix,
                scene,
                axis_size * 0.72 if selected else axis_size * 0.40,
                joint_name=f"A2:{robot_body}" if selected else None,
            )

    # A2 XML 中的两个 site 就是脚底接触点。绿色球低于地面代表穿地。
    for site_name in ("site_l_foot", "site_r_foot"):
        site_id = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_SITE, site_name)
        add_sphere(
            scene,
            viewer.data.site_xpos[site_id],
            radius=0.025,
            color=(0.1, 1.0, 0.1, 1.0),
            label=site_name,
        )


def print_frame_status(
    frame_index,
    total_frames,
    fps,
    retargeter,
    viewer,
    selected_human_body,
    selected_robot_body,
    qpos,
):
    """在终端输出当前帧的重点映射误差、脚底高度和关节限位。"""
    target_position, target_quaternion = retargeter.scaled_human_data[
        selected_human_body
    ]
    robot_position, _, robot_quaternion = get_robot_body_pose(
        viewer, selected_robot_body
    )

    position_error = np.linalg.norm(robot_position - target_position)
    target_rotation = Rotation.from_quat(target_quaternion, scalar_first=True)
    robot_rotation = Rotation.from_quat(robot_quaternion, scalar_first=True)
    rotation_error = np.degrees(
        (target_rotation.inv() * robot_rotation).magnitude()
    )

    left_site = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_SITE, "site_l_foot")
    right_site = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_SITE, "site_r_foot")
    left_sole_z = viewer.data.site_xpos[left_site, 2]
    right_sole_z = viewer.data.site_xpos[right_site, 2]

    lower_limits = viewer.model.jnt_range[1:, 0]
    upper_limits = viewer.model.jnt_range[1:, 1]
    joint_positions = qpos[7:]
    at_limit = np.where(
        (joint_positions <= lower_limits + 1e-4)
        | (joint_positions >= upper_limits - 1e-4)
    )[0]
    limit_names = [viewer.model.joint(i + 1).name for i in at_limit]

    print("\n" + "=" * 72)
    print(
        f"帧 {frame_index}/{total_frames - 1} | "
        f"时间 {frame_index / fps:.3f} 秒 | "
        f"映射 {selected_human_body} -> {selected_robot_body}"
    )
    print(f"重点位置误差：{position_error:.6f} m")
    print(f"重点旋转误差：{rotation_error:.3f} deg")
    print(f"左/右脚底高度：{left_sole_z:.5f} / {right_sole_z:.5f} m")
    print(f"IK 总误差 table1/table2：{retargeter.error1():.6f} / {retargeter.error2():.6f}")
    print(f"触碰关节限位：{limit_names if limit_names else '无'}")
    print(f"A2 12 关节角/rad：{np.array2string(joint_positions, precision=3)}")


def parse_args():
    parser = argparse.ArgumentParser(description="逐帧查看 LAFAN1 到 Taixi A2 的 IK 映射")
    parser.add_argument("--bvh-file", default="lafan1/walk1_subject1.bvh")
    parser.add_argument("--frame", type=int, default=0, help="起始帧，从 0 开始")
    parser.add_argument(
        "--human-body",
        default="LeftFootMod",
        help="重点高亮的人体部位，例如 Hips、LeftLeg、LeftFootMod",
    )
    parser.add_argument("--play", action="store_true", help="启动后立即播放；默认暂停")
    parser.add_argument(
        "--all-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示所有映射坐标轴；使用 --no-all-axes 只看重点和 Hips",
    )
    parser.add_argument("--axis-size", type=float, default=0.24, help="重点坐标轴长度/米")
    parser.add_argument("--fps", type=float, default=30.0, help="播放速度")
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="相机是否跟随 A2 的 base_link；使用 --no-follow-camera 关闭跟随",
    )
    parser.add_argument(
        "--offset-to-ground",
        action="store_true",
        help="启用 GMR 自动地面偏移（建议先关闭，便于检查原配置）",
    )
    parser.add_argument(
        "--opaque-robot",
        action="store_true",
        help="显示不透明 A2；默认半透明，方便观察目标",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames, human_height = load_bvh_file(args.bvh_file, format="lafan1")

    if not 0 <= args.frame < len(frames):
        raise ValueError(f"--frame 应在 0 到 {len(frames) - 1} 之间")

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="taixi_a2",
        actual_human_height=human_height,
        verbose=False,
    )
    pairs = get_mapping_pairs(retargeter)
    pair_dict = dict(pairs)
    if args.human_body not in pair_dict:
        raise ValueError(
            f"{args.human_body!r} 没有映射到 A2。可用名称：{list(pair_dict)}"
        )
    selected_robot_body = pair_dict[args.human_body]

    # 键盘回调和主循环共享这些状态。
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

    print("黄色/橙色=人体目标，青色=A2 Link，紫线=映射误差，绿色=脚底")
    print("操作：左右方向键逐帧，空格播放/暂停，R 回到起始帧")
    print(f"重点映射：{args.human_body} -> {selected_robot_body}")

    viewer = RobotMotionViewer(
        robot_type="taixi_a2",
        motion_fps=args.fps,
        transparent_robot=0 if args.opaque_robot else 1,
        keyboard_callback=on_key,
    )

    last_solved_frame = None
    solved_qpos = None
    human_targets = None

    # 如果从非零帧启动，按 0→起始帧顺序预热 IK，避免从零姿态直接跳转
    # 到远处帧造成很大的假误差。
    if args.frame > 0:
        print(f"正在按顺序预热 IK：0 -> {args.frame} ...")
        for warmup_frame in range(args.frame + 1):
            solved_qpos = retargeter.retarget(
                frames[warmup_frame], offset_to_ground=args.offset_to_ground
            )
        human_targets = retargeter.scaled_human_data
        # 保持为 None，让主循环再求解并打印一次起始帧的完整状态。
        last_solved_frame = None

    try:
        while viewer.viewer.is_running():
            loop_start = time.perf_counter()
            frame_index = state["frame"]

            # 只有帧号变化时才运行一次 IK；暂停后画面不会继续偷偷迭代。
            if frame_index != last_solved_frame:
                solved_qpos = retargeter.retarget(
                    frames[frame_index], offset_to_ground=args.offset_to_ground
                )
                human_targets = retargeter.scaled_human_data
                last_solved_frame = frame_index

                # 先把 qpos 送进显示模型，计算正运动学，再打印实际误差。
                viewer.data.qpos[:] = solved_qpos
                mj.mj_forward(viewer.model, viewer.data)
                if not state["playing"] or frame_index % max(int(args.fps), 1) == 0:
                    print_frame_status(
                        frame_index,
                        len(frames),
                        args.fps,
                        retargeter,
                        viewer,
                        args.human_body,
                        selected_robot_body,
                        solved_qpos,
                    )

            # 起始帧为 0 时，第一次循环会在上面的分支中生成 solved_qpos。
            viewer.data.qpos[:] = solved_qpos
            mj.mj_forward(viewer.model, viewer.data)

            # 跟随模式每帧把相机观察中心移动到机器人 base_link。
            # 固定模式完全不改相机，用户可用鼠标自由选择世界视角。
            if args.follow_camera:
                base_id = mj.mj_name2id(
                    viewer.model, mj.mjtObj.mjOBJ_BODY, "base_link"
                )
                viewer.viewer.cam.lookat = viewer.data.xpos[base_id]
                viewer.viewer.cam.distance = 3.0
                viewer.viewer.cam.elevation = -12

            draw_current_frame(
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
