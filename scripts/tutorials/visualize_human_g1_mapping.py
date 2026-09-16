"""在 MuJoCo 中同时显示人体目标与 Unitree G1，直观理解 IK 映射。

这个脚本的核心数据流：
    1. 从 BVH 读取某一帧人体动作字典；
    2. GMR 对人体位置进行尺寸缩放，并用配置中的旋转偏置校正坐标轴；
    3. IK 求出 G1 的根位姿和 29 个关节角 qpos；
    4. 同时绘制“人体目标位姿”和“机器人实际位姿”；
    5. 用紫线连接每一对映射，紫线越长说明位置误差越大。

注意：这里画出的黄色人体不是原始 BVH 骨架，而是经过 GMR 缩放和坐标
校正、真正送给 IK 求解器的目标。这样才能和 G1 Link 的实际位姿直接比较。

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

# GLFW 提供空格、方向键等键盘按键常量。
import glfw
# mj 是 MuJoCo 的 Python API；RobotMotionViewer 内部也使用同一个 API。
import mujoco as mj
import numpy as np
# SciPy Rotation 用于把 GMR 的 wxyz 四元数转换成 3×3 旋转矩阵。
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
from general_motion_retargeting.robot_motion_viewer import draw_frame
from general_motion_retargeting.utils.lafan1 import load_bvh_file


# 缩放后的 GMR 目标只保留参与 IK 的关键部位，因此不能使用原始 BVH 的
# 完整父子树。这里手工定义一棵“关键点骨架”，每个元组是 (父部位, 子部位)。
# 这些连线只用于显示，不参与 IK 计算。
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
    """向 MuJoCo 的临时场景添加一个球。

    参数：
        viewer: mujoco.viewer.Handle，即实际显示窗口。
        position: 球心的世界坐标 [x, y, z]，单位为米。
        radius: 球半径，单位为米。
        color: RGBA 颜色，每个分量范围为 0~1。
        label: 可选标签；只给重点映射加标签，避免画面太拥挤。

    user_scn 是用户自定义场景。这里添加的球只负责可视化，不会参与
    碰撞、动力学或 IK 求解。
    """
    # ngeom 指向下一个空闲的自定义几何体槽位。
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=position,
        # 球是旋转对称的，单位旋转矩阵即可。
        mat=np.eye(3).flatten(),
        rgba=np.asarray(color, dtype=float),
    )
    if label is not None:
        geom.label = label
    # 告诉 MuJoCo：当前帧的自定义几何体数量增加了一个。
    viewer.user_scn.ngeom += 1


def add_line(viewer, start, end, width, color):
    """用胶囊体画一条从 start 到 end 的三维连线。

    黄色线用作人体骨架，紫色线用作人体目标到机器人 Link 的映射线。
    mjv_connector 会自动计算连线的中心、长度和朝向。
    """
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
    """从第二阶段 IK 表提取 (人体部位, 机器人 Body/Link) 对。

    原 JSON 的结构是：
        "left_wrist_yaw_link": ["LeftHand", 位置权重, 旋转权重, ...]

    为了绘图方便，这里把顺序转换成：
        ("LeftHand", "left_wrist_yaw_link")

    两阶段使用相同的身体对应关系，第二阶段包含最终要比较的位置权重，
    因此本示例从 ik_match_table2 读取。
    """
    return [
        (entry[0], robot_body)
        for robot_body, entry in retargeter.ik_match_table2.items()
    ]


def robot_body_pose(viewer, robot_body):
    """读取某个 G1 Body/Link 求解后的世界位置和世界旋转矩阵。

    data.xpos 的一行是 [x, y, z]；data.xmat 的一行是展平的 3×3 矩阵。
    copy() 可以避免后续 mj_forward 更新数据时改变已经取出的数组视图。
    """
    # MuJoCo 内部使用整数 ID，所以先由字符串名称查找 body ID。
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
    """绘制当前帧的人体目标、G1 Link、映射线和坐标轴。

    human_targets 的结构仍是人体动作字典：
        {人体名称: [世界位置 xyz, 世界旋转四元数 wxyz]}

    但它来自 retargeter.scaled_human_data，已经完成尺寸缩放、地面偏置和
    各部位旋转偏置校正，正是 IK FrameTask 使用的目标。
    """
    # 自定义几何体每帧重新生成。若不清零，球、线和箭头会不断累积。
    viewer.viewer.user_scn.ngeom = 0
    scene = viewer.viewer

    # 1. 人体目标骨架。
    for parent, child in TARGET_SKELETON_EDGES:
        add_line(
            scene,
            # 每个字典值的第 0 项是世界位置，第 1 项是世界旋转。
            human_targets[parent][0],
            human_targets[child][0],
            width=0.011,
            color=(1.0, 0.72, 0.05, 0.9),
        )

    # 2. 每个人体目标与对应 G1 Link 的位置、坐标轴及误差连线。
    for human_body, robot_body in pairs:
        # target_* 是 IK 希望机器人到达的位姿。
        target_position, target_quaternion = human_targets[human_body]
        # robot_* 是用当前 qpos 做正运动学后真正到达的位姿。
        robot_position, robot_matrix = robot_body_pose(viewer, robot_body)
        selected = human_body == selected_human_body

        # 橙色球表示人体 IK 目标。重点部位使用更大的球并显示名称。
        add_sphere(
            scene,
            target_position,
            radius=0.036 if selected else 0.022,
            color=(1.0, 0.25, 0.05, 1.0),
            label=f"Human target: {human_body}" if selected else None,
        )
        # 青色球表示机器人对应 Link 的实际世界位置。
        add_sphere(
            scene,
            robot_position,
            radius=0.032 if selected else 0.018,
            color=(0.0, 0.9, 1.0, 1.0),
            label=f"G1 link: {robot_body}" if selected else None,
        )
        # 紫线连接目标点和实际点，因此线段长度就是位置误差的直观表示。
        add_line(
            scene,
            target_position,
            robot_position,
            width=0.012 if selected else 0.004,
            color=(1.0, 0.0, 0.9, 1.0) if selected else (0.65, 0.2, 1.0, 0.55),
        )

        # 关闭 all_axes 时，仍保留重点部位和根节点 Hips 的坐标轴。
        if all_axes or selected or human_body == "Hips":
            # GMR 四元数顺序为 wxyz，而 SciPy 通常默认 xyzw，所以必须设置
            # scalar_first=True，否则得到的坐标轴方向会完全错误。
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
    """定义命令行参数并返回用户选择。"""
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
    """读取数据、运行逐帧 IK，并维护 MuJoCo 可视化主循环。"""
    args = parse_args()

    # frames 是一个列表；frames[i] 是第 i 帧的人体动作字典。
    # human_height 会参与配置中的人体尺寸缩放。
    frames, human_height = load_bvh_file(args.bvh_file, format="lafan1")
    if not 0 <= args.frame < len(frames):
        raise ValueError(f"--frame 应在 0 到 {len(frames) - 1} 之间")

    # retargeter 持有另一份 G1 MuJoCo 模型，专门用于 Mink IK 求解。
    # viewer 之后会创建第二份模型，专门负责显示；两者通过 qpos 同步。
    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        verbose=False,
    )
    # 例如：("LeftHand", "left_wrist_yaw_link")。
    pairs = mapping_pairs(retargeter)
    available_human_bodies = [human_body for human_body, _ in pairs]
    if args.human_body not in available_human_bodies:
        raise ValueError(
            f"{args.human_body!r} 没有映射到 G1。可用名称：{available_human_bodies}"
        )

    # 键盘回调和绘制循环都需要读写这些值。使用字典后，内部函数 on_key
    # 可以直接修改状态，无需使用多个 nonlocal 变量。
    state = {
        "frame": args.frame,
        "start_frame": args.frame,
        "playing": args.play,
    }

    def on_key(keycode):
        """MuJoCo 窗口的键盘回调。按键后只更新状态，IK 在主循环中执行。"""
        if keycode == glfw.KEY_SPACE:
            # 空格在播放和暂停之间切换。
            state["playing"] = not state["playing"]
        elif keycode == glfw.KEY_RIGHT:
            # 手动逐帧时自动暂停，避免帧号继续增长。
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

    # RobotMotionViewer 加载 G1 XML 并创建被动 MuJoCo 窗口。
    # transparent_robot=1 只改变显示透明度，不影响模型的正运动学结果。
    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        motion_fps=args.fps,
        transparent_robot=0 if args.opaque_robot else 1,
        keyboard_callback=on_key,
    )

    # 缓存最近一次求解结果。暂停画面时不应对同一帧反复运行 IK，否则
    # 每次迭代都可能让机器人继续微调，看起来像暂停后仍在移动。
    last_solved_frame = None
    solved_qpos = None
    human_targets = None
    try:
        # 只要用户没有关闭 MuJoCo 窗口，就不断更新场景。
        while viewer.viewer.is_running():
            # 记录循环起点，用于在末尾把速度限制到 args.fps。
            loop_start = time.perf_counter()
            frame_index = state["frame"]

            # 暂停时不重复求解同一帧，保证画面稳定。
            if frame_index != last_solved_frame:
                # solved_qpos 的格式：根位置 3 + 根四元数 4 + G1 关节角 29。
                solved_qpos = retargeter.retarget(frames[frame_index])
                # 这是实际送入 IK 的人体目标，不是未经处理的原始 BVH 数据。
                human_targets = retargeter.scaled_human_data
                last_solved_frame = frame_index

            # 把 IK 模型算出的 qpos 复制到显示模型。
            viewer.data.qpos[:] = solved_qpos
            # qpos 改变后执行正运动学，更新所有 G1 Link 的 xpos/xmat。
            mj.mj_forward(viewer.model, viewer.data)

            # 相机始终看向 G1 骨盆，机器人行走时相机会跟随。
            pelvis_id = mj.mj_name2id(viewer.model, mj.mjtObj.mjOBJ_BODY, "pelvis")
            viewer.viewer.cam.lookat = viewer.data.xpos[pelvis_id]
            viewer.viewer.cam.distance = 2.2
            viewer.viewer.cam.elevation = -10

            # 在 G1 网格模型之上叠加人体目标、坐标轴和映射误差线。
            draw_mapping_scene(
                viewer,
                human_targets,
                pairs,
                args.human_body,
                args.axis_size,
                args.all_axes,
            )
            # 把 data 和 user_scn 中的修改提交给 MuJoCo 窗口。
            viewer.viewer.sync()

            if state["playing"]:
                # 到达最后一帧后从第 0 帧循环播放。
                state["frame"] = (state["frame"] + 1) % len(frames)

            # 主动等待剩余帧时间，使动画速度接近 --fps。
            remaining = 1.0 / args.fps - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        # 无论正常关闭还是发生异常，都释放 MuJoCo 窗口资源。
        viewer.close()


if __name__ == "__main__":
    main()
