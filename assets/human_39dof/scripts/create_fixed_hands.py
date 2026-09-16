#!/usr/bin/env python3
"""Derive a separate fixed-finger model from the full humanoid export."""
import argparse,json,shutil,xml.etree.ElementTree as E
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOOT_COLLISION_LINKS={"LeftFootLink","LeftForefootLink","RightFootLink","RightForefootLink"}
FOOT_COLLISION_BOXES={
    "LeftFootLink": {"pos":"0.0175 0 -0.11", "urdf_size":"0.165 0.09 0.02", "mj_size":"0.0825 0.045 0.01"},
    "RightFootLink":{"pos":"0.0175 0 -0.11", "urdf_size":"0.165 0.09 0.02", "mj_size":"0.0825 0.045 0.01"},
    "LeftForefootLink": {"pos":"0.0425 0 -0.025", "urdf_size":"0.065 0.09 0.02", "mj_size":"0.0325 0.045 0.01"},
    "RightForefootLink":{"pos":"0.0425 0 -0.025", "urdf_size":"0.065 0.09 0.02", "mj_size":"0.0325 0.045 0.01"},
}
COLLISION_BOXES={
    **FOOT_COLLISION_BOXES,
    # 胸腔视觉网格约为 0.217 x 0.373 x 0.287 m；碰撞盒稍微内缩，
    # 避免肩部和腰部转动时因包围盒过大而过早发生接触。
    "ChestLink":{"pos":"-0.0035 0 0.1435", "urdf_size":"0.18 0.32 0.24", "mj_size":"0.09 0.16 0.12"},
}
ACTIVE_COLLISION_LINKS=set(COLLISION_BOXES)
def save(tree,path):
    E.indent(tree,space='  ');tree.write(path,encoding='utf-8',xml_declaration=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT.parent/'human_robot_export');args=ap.parse_args();src=args.source.resolve()
    assert src!=ROOT
    for folder in ['human_description','mujoco']:
        shutil.copytree(src/folder,ROOT/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['simulate.py','requirements.txt']:
        shutil.copy2(src/'scripts'/name,ROOT/'scripts'/name)
    urdf=ROOT/'human_description/urdf/human.urdf';tree=E.parse(urdf);robot=tree.getroot()
    robot.set('name','human_70kg_fixed_hands')
    fixed=[]
    for joint in robot.findall('joint'):
        name=joint.get('name')
        if 'Finger' in name or 'Thumb' in name:
            assert joint.get('type')=='revolute'
            fixed.append(name);joint.set('type','fixed')
            for child in list(joint):
                if child.tag in ['axis','limit','dynamics','mimic','safety_controller','calibration']:joint.remove(child)
    assert len(fixed)==10
    # 后脚板和前脚板分别属于各自 link，因此前脚板会随 ForefootJoint 旋转。
    # 两块脚板在零位时沿 X 方向留 0.02 m 空隙，避免转动时在关节处互相挤压。
    for link in robot.findall('link'):
        name=link.get('name')
        if name not in ACTIVE_COLLISION_LINKS:
            for collision in link.findall('collision'):
                link.remove(collision)
        else:
            collisions=link.findall('collision');assert len(collisions)==1
            collision=collisions[0];box=COLLISION_BOXES[name]
            origin=collision.find('origin')
            if origin is None:origin=E.SubElement(collision,'origin')
            origin.set('xyz',box['pos']);origin.set('rpy','0 0 0')
            geometry=collision.find('geometry');assert geometry is not None
            for shape in list(geometry):geometry.remove(shape)
            E.SubElement(geometry,'box',{'size':box['urdf_size']})
    save(tree,urdf)
    for variant in ['human.xml','human_fixed.xml']:
        tree=E.parse(ROOT/'mujoco'/variant);model=tree.getroot();model.set('model',model.get('model')+'_fixed_hands')
        for body in model.findall('.//body'):
            for joint in body.findall('joint'):
                if joint.get('name') in fixed:body.remove(joint)
            body_name=body.get('name')
            if body_name not in ACTIVE_COLLISION_LINKS:
                for geom in list(body.findall('geom')):
                    if geom.get('name','').endswith('_collision'):body.remove(geom)
            else:
                collision_geoms=[g for g in body.findall('geom') if g.get('name','').endswith('_collision')]
                assert len(collision_geoms)==1
                geom=collision_geoms[0];box=COLLISION_BOXES[body_name]
                geom.set('type','box');geom.set('pos',box['pos']);geom.set('size',box['mj_size'])
                geom.attrib.pop('mesh',None)
        asset=model.find('asset')
        for mesh in list(asset.findall('mesh')):
            name=mesh.get('name','')
            if name.endswith('_collision'):asset.remove(mesh)
        actuator=model.find('actuator');old=list(actuator);keep=[i for i,a in enumerate(old) if a.get('joint') not in fixed]
        for a in old:
            if a.get('joint') in fixed:actuator.remove(a)
        for key in model.findall('./keyframe/key'):
            qpos=key.get('qpos','').split();ctrl=key.get('ctrl','').split();offset=7 if variant=='human.xml' else 0
            assert len(qpos)==offset+49 and len(ctrl)==49
            key.set('qpos',' '.join(qpos[:offset]+[qpos[offset+i] for i in keep]))
            key.set('ctrl',' '.join(ctrl[i] for i in keep))
        save(tree,ROOT/'mujoco'/variant)
    config=ROOT/'human_description/config/actuation_defaults.json';data=json.loads(config.read_text())
    for name in fixed:del data['joints'][name]
    config.write_text(json.dumps(data,indent=2)+'\n')
    # A distinct ROS package can be installed alongside the original.
    for relative in ['package.xml','CMakeLists.txt','launch/display.launch.py']:
        p=ROOT/'human_description'/relative;p.write_text(p.read_text().replace('human_description','human_fixed_hands_description'))
    (ROOT/'variant.json').write_text(json.dumps({'source_export':src.name,'coordinate_convention':'X forward, Y left, Z up','mass_kg':70,'active_joints':39,'fixed_finger_joints':fixed,'wrist_dofs_per_side':2,'active_collision_links':sorted(ACTIVE_COLLISION_LINKS),'collision_shape':'box_chest_and_split_box_foot_plates','forefoot_joint_gap_m':0.02,'collision_boxes':COLLISION_BOXES,'fixed_pose':'all finger/thumb angles zero','visual_geometry_and_mass_preserved':True},indent=2)+'\n')
    print('Created fixed-hand variant: 39 active joints, 10 fixed finger joints, 70 kg.')
if __name__=='__main__':main()
