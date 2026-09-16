#!/usr/bin/env python3
"""独立地将 LAFAN1 BVH 重映射到 human_39dof，并可预览/导出 GMR PKL。

本程序不会修改 params.py：它只在当前 Python 进程内临时注册模型和 IK 配置。
默认使用首帧自动标定各骨段的局部朝向，并加入低权重姿态正则，减少 39 DoF
模型中腰、脊柱、肩胛等冗余关节的漂移。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
from general_motion_retargeting.params import (
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ROBOT_KEY = "human_39dof_standalone"
MODEL_PATH = REPO_ROOT / "assets" / "human_39dof" / "mujoco" / "human.xml"
CONFIG_PATH = HERE / "bvh_lafan1_to_human_39dof.json"

# 由模型网格包围盒得到的脚底局部点。用于输出前仅向上平移自由根，避免穿地；
# 不修改原始 human.xml，也不会把跳跃动作强行拉回地面。
SOLE_POINTS = (
    ("LeftFootLink", np.array([0.0, 0.0, -0.120])),
    ("LeftForefootLink", np.array([0.050, 0.0, -0.035])),
    ("RightFootLink", np.array([0.0, 0.0, -0.120])),
    ("RightForefootLink", np.array([0.050, 0.0, -0.035])),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-file", default="lafan1/walk1_subject1.bvh")
    parser.add_argument("--start-frame", type=int, default=0, help="输出起始帧（包含）")
    parser.add_argument("--end-frame", type=int, default=None, help="输出终止帧（包含；默认到结尾）")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("outputs/human_39dof_lafan1.pkl"),
        help="输出 PKL 路径",
    )
    parser.add_argument("--no-save", action="store_true", help="只预览，不保存 PKL")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否打开 MuJoCo 预览（默认开启）",
    )
    parser.add_argument("--rate-limit", action="store_true", help="按 --fps 实时预览")
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="预览相机是否跟随骨盆",
    )
    parser.add_argument(
        "--calibration-frame",
        type=int,
        default=0,
        help="骨段朝向自动标定帧（默认 0）",
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=0.5,
        help="零姿态正则权重；越大越不容易出现冗余关节漂移",
    )
    parser.add_argument(
        "--initial-iterations",
        type=int,
        default=15,
        help="正式处理前重复求解第 0 帧的次数，用于消除冷启动误差",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.003,
        help="脚底最小离地高度/米；只在穿地时向上平移根节点",
    )
    parser.add_argument(
        "--ground-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否根据模型真实脚底点进行穿地修正",
    )
    return parser.parse_args()


def register_standalone_model() -> None:
    """仅修改当前进程内的参数字典，不写入任何原有文件。"""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"IK 配置不存在：{CONFIG_PATH}")

    ROBOT_XML_DICT[ROBOT_KEY] = MODEL_PATH
    IK_CONFIG_DICT.setdefault("bvh_lafan1", {})[ROBOT_KEY] = CONFIG_PATH
    ROBOT_BASE_DICT[ROBOT_KEY] = "PelvisLink"
    VIEWER_CAM_DISTANCE_DICT[ROBOT_KEY] = 3.0


def calibrate_orientation_offsets(
    retargeter: GeneralMotionRetargeting,
    calibration_human_frame: dict,
) -> None:
    """让标定帧的骨段相对朝向对应模型零姿态，保留人体根节点朝向。"""
    neutral_data = mj.MjData(retargeter.model)
    neutral_data.qpos[:] = retargeter.model.qpos0
    mj.mj_forward(retargeter.model, neutral_data)

    root_body_id = mj.mj_name2id(
        retargeter.model, mj.mjtObj.mjOBJ_BODY, retargeter.robot_root_name
    )
    neutral_root = Rotation.from_quat(
        neutral_data.xquat[root_body_id], scalar_first=True
    )

    human_root_rotation = Rotation.from_quat(
        calibration_human_frame[retargeter.human_root_name][1], scalar_first=True
    )
    desired_root = human_root_rotation * retargeter.rot_offsets1[
        retargeter.human_root_name
    ]

    for match_table, rotation_offsets in (
        (retargeter.ik_match_table1, retargeter.rot_offsets1),
        (retargeter.ik_match_table2, retargeter.rot_offsets2),
    ):
        for robot_body, entry in match_table.items():
            human_body = entry[0]
            if human_body == retargeter.human_root_name:
                continue

            robot_body_id = mj.mj_name2id(
                retargeter.model, mj.mjtObj.mjOBJ_BODY, robot_body
            )
            neutral_body = Rotation.from_quat(
                neutral_data.xquat[robot_body_id], scalar_first=True
            )
            neutral_relative_rotation = neutral_root.inv() * neutral_body
            desired_body = desired_root * neutral_relative_rotation

            human_body_rotation = Rotation.from_quat(
                calibration_human_frame[human_body][1], scalar_first=True
            )
            rotation_offsets[human_body] = human_body_rotation.inv() * desired_body


def add_posture_regularization(
    retargeter: GeneralMotionRetargeting,
    posture_cost: float,
) -> None:
    """向两个求解阶段加入零姿态正则，稳定未被充分观测的关节。"""
    if posture_cost <= 0:
        return

    costs = np.full(retargeter.model.nv, posture_cost, dtype=float)
    # LAFAN1 没有独立肩胛观测，避免求解器为了追手的位置把肩胛推到限位。
    for joint_name in (
        "RightScapulaProtractionJoint",
        "RightScapulaElevationJoint",
        "LeftScapulaProtractionJoint",
        "LeftScapulaElevationJoint",
    ):
        joint_id = mj.mj_name2id(
            retargeter.model, mj.mjtObj.mjOBJ_JOINT, joint_name
        )
        costs[retargeter.model.jnt_dofadr[joint_id]] = posture_cost * 10.0
    # 手部仅追踪位置。腕关节不会改变 HandLink 原点的位置，因而属于数值
    # 零空间；强正则可防止它们无意义地漂移到限位。
    for joint_name in (
        "RightWristFlexJoint",
        "RightWristSideJoint",
        "LeftWristFlexJoint",
        "LeftWristSideJoint",
    ):
        joint_id = mj.mj_name2id(
            retargeter.model, mj.mjtObj.mjOBJ_JOINT, joint_name
        )
        costs[retargeter.model.jnt_dofadr[joint_id]] = posture_cost * 50.0
    # 腰和脊柱共同承担躯干运动，略强正则可减少二者互相补偿。
    for joint_name in (
        "WaistFlexJoint",
        "WaistSideJoint",
        "WaistTwistJoint",
        "SpineFlexJoint",
        "SpineSideJoint",
        "SpineTwistJoint",
    ):
        joint_id = mj.mj_name2id(
            retargeter.model, mj.mjtObj.mjOBJ_JOINT, joint_name
        )
        costs[retargeter.model.jnt_dofadr[joint_id]] = posture_cost * 2.0

    for tasks in (retargeter.tasks1, retargeter.tasks2):
        task = mink.PostureTask(
            retargeter.model,
            cost=costs,
            lm_damping=1.0,
        )
        task.set_target(retargeter.model.qpos0.copy())
        tasks.append(task)


def correct_ground_penetration(
    retargeter: GeneralMotionRetargeting,
    qpos: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, float]:
    """若任一脚底低于 clearance，只向上平移自由根，并返回修正量。"""
    configuration = retargeter.configuration
    configuration.data.qpos[:] = qpos
    mj.mj_forward(retargeter.model, configuration.data)

    minimum_z = np.inf
    for body_name, local_point in SOLE_POINTS:
        body_id = mj.mj_name2id(
            retargeter.model, mj.mjtObj.mjOBJ_BODY, body_name
        )
        world_point = (
            configuration.data.xpos[body_id]
            + configuration.data.xmat[body_id].reshape(3, 3) @ local_point
        )
        minimum_z = min(minimum_z, float(world_point[2]))

    correction = max(clearance - minimum_z, 0.0)
    corrected = qpos.copy()
    corrected[2] += correction
    return corrected, correction


def validate_args(args: argparse.Namespace, frame_count: int) -> tuple[int, int]:
    if args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if args.posture_cost < 0:
        raise ValueError("--posture-cost 不能小于 0")
    if args.initial_iterations < 0:
        raise ValueError("--initial-iterations 不能小于 0")
    if args.ground_clearance < 0:
        raise ValueError("--ground-clearance 不能小于 0")
    if not 0 <= args.start_frame < frame_count:
        raise ValueError(f"--start-frame 应在 0 到 {frame_count - 1} 之间")

    end_frame = frame_count - 1 if args.end_frame is None else args.end_frame
    if not args.start_frame <= end_frame < frame_count:
        raise ValueError(
            f"--end-frame 应在 {args.start_frame} 到 {frame_count - 1} 之间"
        )
    if not 0 <= args.calibration_frame < frame_count:
        raise ValueError(f"--calibration-frame 应在 0 到 {frame_count - 1} 之间")
    return args.start_frame, end_frame


def save_motion(path: Path, fps: float, qpos_frames: list[np.ndarray]) -> None:
    if not qpos_frames:
        raise RuntimeError("没有可保存的重映射帧")
    qpos = np.asarray(qpos_frames)
    motion_data = {
        "fps": fps,
        "root_pos": qpos[:, :3],
        # MuJoCo/GMR 内部为 wxyz，PKL 沿用项目既有的 xyzw 格式。
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(motion_data, file)
    print(f"已保存 {len(qpos_frames)} 帧到：{path.resolve()}")


def main() -> None:
    args = parse_args()
    register_standalone_model()

    frames, human_height = load_bvh_file(args.bvh_file, format="lafan1")
    start_frame, end_frame = validate_args(args, len(frames))

    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot=ROBOT_KEY,
        actual_human_height=human_height,
        verbose=False,
    )
    calibrate_orientation_offsets(retargeter, frames[args.calibration_frame])
    add_posture_regularization(retargeter, args.posture_cost)

    if args.initial_iterations:
        print(f"初始姿态收敛：重复求解第 0 帧 {args.initial_iterations} 次")
        for _ in range(args.initial_iterations):
            retargeter.retarget(frames[0])

    viewer = None
    if args.viewer:
        viewer = RobotMotionViewer(
            robot_type=ROBOT_KEY,
            motion_fps=args.fps,
            transparent_robot=0,
        )

    # GMR 使用上一帧作为下一帧的初值。从中间开始导出时仍按顺序预热，
    # 避免直接从零姿态跳到远处帧而得到错误的局部解。
    if start_frame > 0:
        print(f"预热 IK：0 -> {start_frame - 1}")
        for frame_index in tqdm(range(start_frame), desc="IK warm-up"):
            retargeter.retarget(frames[frame_index])

    qpos_frames: list[np.ndarray] = []
    ground_corrections: list[float] = []
    try:
        for frame_index in tqdm(
            range(start_frame, end_frame + 1), desc="Retargeting"
        ):
            qpos = retargeter.retarget(frames[frame_index])
            correction = 0.0
            if args.ground_correction:
                qpos, correction = correct_ground_penetration(
                    retargeter, qpos, args.ground_clearance
                )
            qpos_frames.append(qpos)
            ground_corrections.append(correction)

            if viewer is not None:
                if not viewer.viewer.is_running():
                    print("MuJoCo 窗口已关闭，停止处理。")
                    break
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=args.follow_camera,
                )
    finally:
        if viewer is not None:
            viewer.close()

    if not args.no_save:
        save_motion(args.save_path, args.fps, qpos_frames)

    if ground_corrections:
        corrections = np.asarray(ground_corrections)
        print(
            "地面修正：最大 "
            f"{corrections.max():.4f} m，"
            f"发生于 {(corrections > 0).sum()}/{len(corrections)} 帧"
        )
    print(
        f"最终 IK 误差 table1/table2："
        f"{retargeter.error1():.6f} / {retargeter.error2():.6f}"
    )


if __name__ == "__main__":
    main()
