"""零基础示例：理解人体部位如何成为机器人的 IK 目标。

默认观察：LeftHand -> left_wrist_yaw_link

运行：
    conda activate env_gmr_0
    python scripts/tutorials/learn_human_robot_mapping.py

切换身体部位或帧：
    python scripts/tutorials/learn_human_robot_mapping.py --human-body LeftFootMod
    python scripts/tutorials/learn_human_robot_mapping.py --frame 100 --human-body Hips
"""

import argparse
import json

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


SOURCE_FORMAT = "bvh_lafan1"
ROBOT_NAME = "unitree_g1"


def format_array(value):
    return np.array2string(np.asarray(value), precision=4, suppress_small=True)


def find_robot_body(match_table, human_body):
    """根据人体名称反查对应的机器人 body/link 名称。"""
    for robot_body, entry in match_table.items():
        if entry[0] == human_body:
            return robot_body
    return None


def print_mapping_table(config):
    print("=== 1. 完整关键部位映射 ===")
    print("人体部位                 -> 机器人 Body/Link")
    print("-" * 65)
    for robot_body, entry in config["ik_match_table2"].items():
        human_body = entry[0]
        print(f"{human_body:<24} -> {robot_body}")


def print_pose(title, position, quaternion):
    euler = Rotation.from_quat(quaternion, scalar_first=True).as_euler(
        "xyz", degrees=True
    )
    print(title)
    print(f"  位置 xyz/米       : {format_array(position)}")
    print(f"  旋转 wxyz         : {format_array(quaternion)}")
    print(f"  欧拉角 xyz/度     : {format_array(euler)}（仅辅助观察）")


def largest_joint_changes(model, initial_qpos, solved_qpos, limit=8):
    """列出 IK 前后变化最大的单自由度关节，不包含浮动根节点。"""
    changes = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        qpos_index = model.jnt_qposadr[joint_id]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        delta = solved_qpos[qpos_index] - initial_qpos[qpos_index]
        changes.append((abs(delta), joint_name, delta, joint_type))

    return sorted(changes, reverse=True)[:limit]


def parse_args():
    parser = argparse.ArgumentParser(description="查看人体部位到机器人 IK 目标的完整过程")
    parser.add_argument("--bvh-file", default="lafan1/walk1_subject1.bvh")
    parser.add_argument("--frame", type=int, default=0, help="观察帧，从 0 开始")
    parser.add_argument(
        "--human-body",
        default="LeftHand",
        help="配置中使用的人体部位，例如 LeftHand、Hips、LeftFootMod",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = IK_CONFIG_DICT[SOURCE_FORMAT][ROBOT_NAME]
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    print_mapping_table(config)

    robot_body = find_robot_body(config["ik_match_table2"], args.human_body)
    if robot_body is None:
        available = [entry[0] for entry in config["ik_match_table2"].values()]
        raise ValueError(f"{args.human_body!r} 没有参与 IK。可用人体部位：{available}")

    frames, actual_human_height = load_bvh_file(args.bvh_file, format="lafan1")
    if not 0 <= args.frame < len(frames):
        raise ValueError(f"--frame 应在 0 到 {len(frames) - 1} 之间")
    human_frame = frames[args.frame]

    table1_entry = config["ik_match_table1"][robot_body]
    table2_entry = config["ik_match_table2"][robot_body]
    _, pos_weight1, rot_weight1, pos_offset, rot_offset = table1_entry
    _, pos_weight2, rot_weight2, _, _ = table2_entry

    base_scale = config["human_scale_table"][args.human_body]
    height_ratio = actual_human_height / config["human_height_assumption"]
    effective_scale = base_scale * height_ratio

    print("\n=== 2. 重点映射的配置 ===")
    print(f"当前帧                  : {args.frame}")
    print(f"人体部位                : {args.human_body}")
    print(f"机器人 Body/Link        : {robot_body}")
    print(f"第一阶段位置/旋转权重   : {pos_weight1} / {rot_weight1}")
    print(f"第二阶段位置/旋转权重   : {pos_weight2} / {rot_weight2}")
    print(f"位置偏置                : {pos_offset}")
    print(f"旋转偏置 wxyz           : {rot_offset}")
    print(
        f"有效位置缩放            : {base_scale} × "
        f"({actual_human_height}/{config['human_height_assumption']}) "
        f"= {effective_scale:.4f}"
    )

    raw_position, raw_quaternion = human_frame[args.human_body]
    print("\n=== 3. 原始人体位姿 ===")
    print_pose("BVH 字典中的全局位姿：", raw_position, raw_quaternion)

    # 初始化会读取机器人 XML 和映射配置。
    retargeter = GeneralMotionRetargeting(
        src_human=SOURCE_FORMAT,
        tgt_robot=ROBOT_NAME,
        actual_human_height=actual_human_height,
        verbose=False,
    )
    initial_qpos = retargeter.configuration.data.qpos.copy()

    # 一次 retarget 会同时处理当前帧的全部映射目标。
    solved_qpos = retargeter.retarget(human_frame)
    target_position, target_quaternion = retargeter.scaled_human_data[args.human_body]

    print("\n=== 4. 缩放和坐标校正后的 IK 目标 ===")
    print_pose("机器人对应 Link 应尽量到达：", target_position, target_quaternion)

    body_id = mj.mj_name2id(
        retargeter.model, mj.mjtObj.mjOBJ_BODY, robot_body
    )
    robot_position = retargeter.configuration.data.xpos[body_id].copy()
    robot_quaternion = retargeter.configuration.data.xquat[body_id].copy()

    print("\n=== 5. IK 求解后的机器人实际位姿 ===")
    print_pose(f"{robot_body} 的实际全局位姿：", robot_position, robot_quaternion)

    position_error = np.linalg.norm(robot_position - target_position)
    target_rotation = Rotation.from_quat(target_quaternion, scalar_first=True)
    robot_rotation = Rotation.from_quat(robot_quaternion, scalar_first=True)
    rotation_error_degrees = np.degrees(
        (target_rotation.inv() * robot_rotation).magnitude()
    )

    print("\n=== 6. 目标与实际值的误差 ===")
    print(f"位置误差                : {position_error:.6f} 米")
    print(f"旋转误差                : {rotation_error_degrees:.4f} 度")
    print("误差不一定为零：IK 还要同时兼顾其他身体目标和机器人的关节限制。")

    print("\n=== 7. 整帧 IK 中变化最大的机器人关节 ===")
    print("注意：这些变化由当前帧的全部人体目标共同产生，不只由所选部位产生。")
    for _, joint_name, delta, joint_type in largest_joint_changes(
        retargeter.model, initial_qpos, solved_qpos
    ):
        if joint_type == mj.mjtJoint.mjJNT_HINGE:
            print(f"{joint_name:<32}: {np.degrees(delta):>9.3f} 度")
        else:
            print(f"{joint_name:<32}: {delta:>9.4f} 米")

    print("\n=== 8. 一句话理解 ===")
    print(
        f"GMR 没有把 {args.human_body} 的角度复制给某个电机；它把人体位姿变成 "
        f"{robot_body} 的目标，再由 IK 联合计算整台机器人的 qpos。"
    )


if __name__ == "__main__":
    main()
