# human_39dof 独立重映射教程

这个目录是独立的 LAFAN1 → `human_39dof` 全身 IK 示例，不修改：

- `general_motion_retargeting/params.py`
- `scripts/bvh_to_robot.py`
- 已有机器人 IK JSON
- `assets/human_39dof` 中的模型文件

程序运行时只在当前 Python 进程中临时注册模型和配置，程序退出后不会留下全局注册。

## 文件

- `bvh_lafan1_to_human_39dof.json`：独立 IK 映射配置。
- `retarget_lafan1_to_human_39dof.py`：重映射、预览、地面修正和 PKL 导出。
- `play_human_39dof_pkl.py`：独立回放已生成的 PKL。

使用浮动基座模型：

```text
assets/human_39dof/mujoco/human.xml
```

不要换成 `human_fixed.xml`，否则输出中没有可移动的自由根节点。

## 先测试 300 帧

```bash
cd /home/libai/09_gmr/GMR

conda run --no-capture-output -n env_gmr_0 \
python scripts/tutorials/human_39dof_retargeting/retarget_lafan1_to_human_39dof.py \
  --bvh-file lafan1/walk1_subject1.bvh \
  --start-frame 0 \
  --end-frame 299 \
  --save-path outputs/human_39dof_walk_test.pkl \
  --rate-limit
```

去掉 `--rate-limit` 会尽可能快地处理。

## 生成完整 PKL

完整动作建议关闭预览：

```bash
conda run --no-capture-output -n env_gmr_0 \
python scripts/tutorials/human_39dof_retargeting/retarget_lafan1_to_human_39dof.py \
  --bvh-file lafan1/walk1_subject1.bvh \
  --no-viewer \
  --save-path outputs/human_39dof_walk_full.pkl
```

## 只预览，不保存

```bash
conda run --no-capture-output -n env_gmr_0 \
python scripts/tutorials/human_39dof_retargeting/retarget_lafan1_to_human_39dof.py \
  --end-frame 299 \
  --no-save \
  --rate-limit
```

## 中间片段

```bash
conda run --no-capture-output -n env_gmr_0 \
python scripts/tutorials/human_39dof_retargeting/retarget_lafan1_to_human_39dof.py \
  --start-frame 600 \
  --end-frame 899 \
  --no-viewer \
  --save-path outputs/human_39dof_frames_600_899.pkl
```

即使从第 600 帧开始，程序也会先用 0～599 帧预热 IK，保证连续求解。

## 关键参数

- `--posture-cost`：姿态正则权重，默认 `0.5`。关节乱动时增大，动作过硬时减小。
- `--initial-iterations`：第 0 帧冷启动收敛次数，默认 `15`。
- `--ground-clearance`：脚底最低高度，默认 `0.003 m`。
- `--no-ground-correction`：关闭脚底穿地修正。
- `--calibration-frame`：局部坐标自动标定帧，默认第 0 帧。
- `--no-follow-camera`：相机不跟随骨盆。
- `--no-viewer`：不开窗口，适合快速生成完整数据。

地面修正只在脚底穿地时向上移动自由根，不会把跳跃动作向下拉回地面。

## 播放生成的 PKL

```bash
conda run --no-capture-output -n env_gmr_0 \
python scripts/tutorials/human_39dof_retargeting/play_human_39dof_pkl.py \
  outputs/human_39dof_walk_full.pkl
```

慢速播放并从第 300 帧开始：

```bash
conda run --no-capture-output -n env_gmr_0 \
python scripts/tutorials/human_39dof_retargeting/play_human_39dof_pkl.py \
  outputs/human_39dof_walk_full.pkl \
  --frame 300 \
  --speed 0.5
```

键盘：空格播放/暂停，左右方向键逐帧，`R` 回到起始帧。

## 当前映射

- `Hips` → `PelvisLink`
- `Spine1` → `TorsoLink`
- `Spine2` → `ChestLink`
- `Head` → `HeadLink`
- 双侧 `Arm / ForeArm / Hand` → 上臂、前臂、手
- 双侧 `UpLeg / Leg / FootMod / Toe` → 大腿、小腿、脚、前脚掌

肩胛关节暂时没有单独人体观测，由姿态正则保持稳定。
