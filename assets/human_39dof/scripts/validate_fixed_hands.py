#!/usr/bin/env python3
import argparse,json,hashlib,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
FOOT_COLLISION_LINKS={"LeftFootLink","LeftForefootLink","RightFootLink","RightForefootLink"}
FOOT_COLLISION_BOXES={
    "LeftFootLink": ([0.0175,0,-0.11],[0.165,0.09,0.02],[0.0825,0.045,0.01]),
    "RightFootLink":([0.0175,0,-0.11],[0.165,0.09,0.02],[0.0825,0.045,0.01]),
    "LeftForefootLink": ([0.0425,0,-0.025],[0.065,0.09,0.02],[0.0325,0.045,0.01]),
    "RightForefootLink":([0.0425,0,-0.025],[0.065,0.09,0.02],[0.0325,0.045,0.01]),
}
COLLISION_BOXES={
    **FOOT_COLLISION_BOXES,
    "ChestLink":([-0.0035,0,0.1435],[0.18,0.32,0.24],[0.09,0.16,0.12]),
}
ACTIVE_COLLISION_LINKS=set(COLLISION_BOXES)
def transform(p,r=None):
    T=np.eye(4);T[:3,3]=p
    if r is not None:T[:3,:3]=r
    return T
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT.parent/'human_robot_export');args=ap.parse_args()
    info=json.loads((ROOT/'variant.json').read_text());fixed=set(info['fixed_finger_joints'])
    urdf=ROOT/'human_description/urdf/human.urdf';robot=E.parse(urdf).getroot();joints=robot.findall('joint')
    active=[j for j in joints if j.get('type')=='revolute']
    assert len(active)==39 and sum(j.get('type')=='fixed' for j in joints)==11
    assert all(j.get('type')=='fixed' for j in joints if j.get('name') in fixed)
    wrists=[j.get('name') for j in active if 'Wrist' in j.get('name')];assert len(wrists)==4
    assert abs(sum(float(m.get('value')) for m in robot.findall('./link/inertial/mass'))-70)<1e-8
    collision_links={link.get('name') for link in robot.findall('link') if link.find('collision') is not None}
    assert collision_links==ACTIVE_COLLISION_LINKS
    for link in robot.findall('link'):
        name=link.get('name')
        if name not in ACTIVE_COLLISION_LINKS:continue
        collision=link.find('collision');pos,urdf_size,_=COLLISION_BOXES[name]
        assert np.allclose(np.fromstring(collision.find('origin').get('xyz'),sep=' '),pos)
        geometry=collision.find('geometry');box=geometry.find('box')
        assert box is not None and len(list(geometry))==1
        assert np.allclose(np.fromstring(box.get('size'),sep=' '),urdf_size)
    for mesh in robot.findall('.//mesh'):
        p=mesh.get('filename');assert p.startswith('../meshes/') and (urdf.parent/p).is_file()
    for p in (ROOT/'human_description/meshes').rglob('*.stl'):
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((args.source/p.relative_to(ROOT)).read_bytes()).digest()
    def fk(angles):
        frames={'base_link':np.eye(4)};todo=joints.copy()
        while todo:
            count=len(todo)
            for j in todo[:]:
                parent=j.find('parent').get('link')
                if parent not in frames:continue
                origin=j.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
                T=transform(xyz,Rotation.from_euler('xyz',rpy).as_matrix())
                if j.get('type')=='revolute':T=T@transform([0,0,0],Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*angles[j.get('name')]).as_matrix())
                frames[j.find('child').get('link')]=frames[parent]@T;todo.remove(j)
            assert len(todo)<count
        return frames
    report={'active_joints':39,'fixed_finger_joints':10,'preserved_wrist_joints':wrists,'total_mass_kg':70,'identical_meshes':60,'active_collision_links':sorted(ACTIVE_COLLISION_LINKS),'collision_shape':'box_chest_and_split_box_foot_plates','forefoot_joint_gap_m':0.02}
    rng=np.random.default_rng(712)
    for variant in ['human.xml','human_fixed.xml']:
        model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco'/variant));data=mujoco.MjData(model)
        old=mujoco.MjModel.from_xml_path(str(args.source/'mujoco'/variant));od=mujoco.MjData(old)
        assert model.nu==39 and model.nv==(45 if variant=='human.xml' else 39)
        assert abs(model.body_mass.sum()-70)<1e-8
        collision_geoms={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) for gid in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) or '').endswith('_collision')}
        assert collision_geoms=={name+'_collision' for name in ACTIVE_COLLISION_LINKS}
        for name,(pos,_,mj_size) in COLLISION_BOXES.items():
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name+'_collision')
            assert model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX
            assert np.allclose(model.geom_pos[gid],pos) and np.allclose(model.geom_size[gid],mj_size)
        xml_root=E.parse(ROOT/'mujoco'/variant).getroot()
        assert not any(mesh.get('name','').endswith('_collision') for mesh in xml_root.findall('./asset/mesh'))
        for name in fixed:assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)==-1
        error=0
        for pose in range(6):
            mujoco.mj_resetData(model,data);mujoco.mj_resetData(old,od);angles={}
            for j in active:
                name=j.get('name');limit=j.find('limit');lo=float(limit.get('lower'));hi=float(limit.get('upper'))
                value=0 if pose==0 else rng.uniform(lo*.65,hi*.65);angles[name]=value
                for m,d in [(model,data),(old,od)]:
                    jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);assert jid>=0
                    assert np.max(abs(m.jnt_range[jid]-[lo,hi]))<1e-9
                    d.qpos[m.jnt_qposadr[jid]]=value
            mujoco.mj_forward(model,data);mujoco.mj_forward(old,od);frames=fk(angles)
            for bid in range(1,model.nbody):
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid);oid=mujoco.mj_name2id(old,mujoco.mjtObj.mjOBJ_BODY,name)
                actual=transform(data.xpos[bid],data.xmat[bid].reshape(3,3))
                expected=transform(od.xpos[oid],od.xmat[oid].reshape(3,3))
                error=max(error,float(np.max(abs(actual-expected))),float(np.max(abs(actual-frames[name]))))
        assert error<1e-8
        mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
        for side in ['Left','Right']:
            rear=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,side+'FootLink_collision')
            front=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,side+'ForefootLink_collision')
            rear_end=data.geom_xpos[rear,0]+model.geom_size[rear,0]
            front_start=data.geom_xpos[front,0]-model.geom_size[front,0]
            assert abs(front_start-rear_end-0.02)<1e-9
            assert abs(data.geom_xpos[rear,2]-model.geom_size[rear,2])<1e-9
            assert abs(data.geom_xpos[front,2]-model.geom_size[front,2])<1e-9
        zero_forefoot_mats={side:data.geom_xmat[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,side+'ForefootLink_collision')].copy() for side in ['Left','Right']}
        rotating_clearances=[]
        for angle in [-1.0471975512,0,0.349065850399]:
            mujoco.mj_resetData(model,data)
            for side in ['Left','Right']:
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,side+'ForefootJoint')
                data.qpos[model.jnt_qposadr[jid]]=angle
            mujoco.mj_forward(model,data)
            for side in ['Left','Right']:
                rear=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,side+'FootLink_collision')
                front=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,side+'ForefootLink_collision')
                rear_radius=np.abs(data.geom_xmat[rear].reshape(3,3)[0])@model.geom_size[rear]
                front_radius=np.abs(data.geom_xmat[front].reshape(3,3)[0])@model.geom_size[front]
                rotating_clearances.append(data.geom_xpos[front,0]-front_radius-data.geom_xpos[rear,0]-rear_radius)
        assert min(rotating_clearances)>0.005
        assert all(not np.allclose(data.geom_xmat[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,side+'ForefootLink_collision')],zero_forefoot_mats[side]) for side in ['Left','Right'])
        mujoco.mj_resetData(model,data)
        for _ in range(2000):
            mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            assert not any(w.number for w in data.warning)
        report[variant]={'nq':model.nq,'nv':model.nv,'actuators':model.nu,'pose_comparisons':6,'max_transform_error':error,'minimum_joint_range_clearance_m':min(rotating_clearances),'physics_steps':2000,'warnings':0}
    (ROOT/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
