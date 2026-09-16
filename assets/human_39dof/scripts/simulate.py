#!/usr/bin/env python3
"""MuJoCo viewer / smoke demo. Motors take torque, not joint position."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixed',action='store_true');ap.add_argument('--hold',action='store_true');ap.add_argument('--viewer',action='store_true');ap.add_argument('--seconds',type=float,default=10);args=ap.parse_args()
    model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco'/('human_fixed.xml' if args.fixed else 'human.xml')))
    data=mujoco.MjData(model)
    config=json.loads((ROOT/'human_description/config/actuation_defaults.json').read_text())['joints']
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in model.actuator_trnid[:,0]]
    def step():
        mujoco.mj_forward(model,data)
        if args.hold:
            for aid,name in enumerate(names):
                jid=int(model.actuator_trnid[aid,0]);qa=model.jnt_qposadr[jid];va=model.jnt_dofadr[jid];c=config[name]
                torque=-c['kp']*data.qpos[qa]-c['kd']*data.qvel[va]+data.qfrc_bias[va]
                data.ctrl[aid]=np.clip(torque,*model.actuator_ctrlrange[aid])
        mujoco.mj_step(model,data)
    if args.viewer:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(model,data) as viewer:
            viewer.cam.lookat[:]=[0,0,.9];viewer.cam.distance=3;viewer.cam.azimuth=185;viewer.cam.elevation=-12
            while viewer.is_running() and data.time<args.seconds:
                start=time.perf_counter();step();viewer.sync();time.sleep(max(0,model.opt.timestep-(time.perf_counter()-start)))
    else:
        while data.time<args.seconds:step()
        assert np.isfinite(data.qpos).all()
        print(f'Simulated {data.time:.3f} s; mass={model.body_mass.sum():.6f} kg; warnings={sum(w.number for w in data.warning)}')
if __name__=='__main__':main()
