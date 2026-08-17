"""零基础示例：看懂 GMR 的人体动作字典。

运行方式：

    conda activate env_gmr_0
    python scripts/tutorials/learn_human_motion_dict.py

也可以观察其他帧或身体部位：

    python scripts/tutorials/learn_human_motion_dict.py --frame 100 --body RightHand
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting.utils.lafan1 import load_bvh_file


def format_vector(vector: np.ndarray) -> str:
    """把数组格式化为容易阅读的小数。"""
    return np.array2string(vector, precision=4, suppress_small=True)


def describe_body(frame: dict, body_name: str, root_name: str = "Hips") -> None:
    """打印一个身体部位的位置、旋转以及它相对根节点的位置。"""
    world_position, world_quaternion = frame[body_name]
    root_position, root_quaternion = frame[root_name]

    # 1. 只把原点移动到 Hips，但坐标轴仍与世界坐标轴平行。
    root_relative_world_axes = world_position - root_position

    # 2. 严格转换到 Hips 自身的局部坐标系。
    # GMR 的四元数顺序是 wxyz，因此要设置 scalar_first=True。
    root_rotation = Rotation.from_quat(root_quaternion, scalar_first=True)
    root_local_position = root_rotation.inv().apply(root_relative_world_axes)

    # 欧拉角更直观，只用于观察；GMR 内部仍使用四元数计算。
    euler_xyz_degrees = Rotation.from_quat(
        world_quaternion, scalar_first=True
    ).as_euler("xyz", degrees=True)

    print(f"\n[{body_name}]")
    print(f"  世界位置 xyz（米）       : {format_vector(world_position)}")
    print(f"  相对 Hips（世界轴方向）  : {format_vector(root_relative_world_axes)}")
    print(f"  相对 Hips（Hips 局部轴） : {format_vector(root_local_position)}")
    print(f"  世界旋转 wxyz            : {format_vector(world_quaternion)}")
    print(f"  四元数长度               : {np.linalg.norm(world_quaternion):.6f}")
    print(f"  仅供观察的欧拉角 xyz/度  : {format_vector(euler_xyz_degrees)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 GMR 人体动作字典")
    parser.add_argument(
        "--bvh-file",
        default="lafan1/walk1_subject1.bvh",
        help="要读取的 LAFAN1 BVH 文件",
    )
    parser.add_argument("--frame", type=int, default=0, help="要观察的帧编号，从 0 开始")
    parser.add_argument("--body", default="LeftHand", help="重点观察的身体部位")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, human_height = load_bvh_file(args.bvh_file, format="lafan1")

    if not 0 <= args.frame < len(frames):
        raise ValueError(f"帧编号应在 0 到 {len(frames) - 1} 之间")

    frame = frames[args.frame]
    if args.body not in frame:
        available = ", ".join(frame.keys())
        raise ValueError(f"没有身体部位 {args.body!r}。可用名称：{available}")

    print("=== 1. 整段动作 ===")
    print(f"BVH 文件       : {args.bvh_file}")
    print(f"总帧数         : {len(frames)}")
    print(f"人体参考身高   : {human_height:.2f} 米")

    print("\n=== 2. 当前帧字典 ===")
    print(f"帧编号         : {args.frame}")
    print(f"Python 类型    : {type(frame).__name__}")
    print(f"身体部位数量   : {len(frame)}")
    print(f"所有字典键     : {list(frame.keys())}")
    print("每个值的结构   : [位置 xyz, 旋转四元数 wxyz]")

    print("\n=== 3. 身体部位数据 ===")
    # Hips 是人体根节点；另外打印用户选择的部位和双脚，便于比较高度。
    body_names = ["Hips", args.body, "LeftFootMod", "RightFootMod"]
    for body_name in dict.fromkeys(body_names):
        describe_body(frame, body_name)

    print("\n=== 4. 坐标系结论 ===")
    print("GMR 此处的位置单位是米，数据经过转换后使用 Z 轴表示高度。")
    print("字典中的位置和旋转都是世界坐标（全局坐标）。")
    print("四元数采用 wxyz 顺序；欧拉角只适合辅助观察。")


if __name__ == "__main__":
    main()
