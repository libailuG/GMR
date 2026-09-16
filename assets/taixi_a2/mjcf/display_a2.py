import mujoco
import mujoco_viewer
import numpy as np
import time
import math

# 加载模型
MODEL_PATH = "a2.xml"
# MODEL_PATH = '/home/libai/00_isaaclab/taixi_centaur01/model/taixi_centaur01/urdf_foot/taixi_centaur01.xml'

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)
viewer.vopt.geomgroup[3] = 1  # 显示碰撞体
# viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1  # 显示执行器力



# 仿真循环
while viewer.is_alive:
    # 持续下发位置指令 (PD 控制器会自动跟踪)
    # data.ctrl[:] = stand_pose

    mujoco.mj_step(model, data)
    viewer.render()
    time.sleep(0.001)

viewer.close()