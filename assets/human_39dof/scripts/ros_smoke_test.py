#!/usr/bin/env python3
"""Run with ROS's /usr/bin/python3 after sourcing the built workspace."""
import os,subprocess,signal,time,json
from pathlib import Path
os.environ.setdefault('ROS_DOMAIN_ID','79');os.environ.setdefault('ROS_LOCALHOST_ONLY','1')
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
ROOT=Path(__file__).resolve().parents[1]
def main():
    proc=subprocess.Popen(['ros2','launch','human_fixed_hands_description','display.launch.py','gui:=false','rviz:=false'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    rclpy.init();node=rclpy.create_node('human_export_verification')
    state={'joints':set(),'dynamic':set(),'static':set()}
    def joint_cb(msg):state['joints'].update(msg.name)
    def tf_cb(msg):state['dynamic'].update(t.child_frame_id for t in msg.transforms)
    def static_cb(msg):state['static'].update(t.child_frame_id for t in msg.transforms)
    node.create_subscription(JointState,'/joint_states',joint_cb,10)
    node.create_subscription(TFMessage,'/tf',tf_cb,10)
    node.create_subscription(TFMessage,'/tf_static',static_cb,QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        end=time.monotonic()+18
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.2)
            if len(state['joints'])==39 and len(state['dynamic'])==39 and len(state['static'])==11:break
        assert len(state['joints'])==39,state
        assert len(state['dynamic'])==39,state
        assert len(state['static'])==11,state
        report={'ros_distro':os.environ.get('ROS_DISTRO'),'joint_states_count':len(state['joints']),'dynamic_tf_count':len(state['dynamic']),'static_tf_count':len(state['static']),'launch_success':True}
        (ROOT/'ros_validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    finally:
        node.destroy_node();rclpy.shutdown();os.killpg(proc.pid,signal.SIGINT)
        try:output,_=proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);output,_=proc.communicate(timeout=5)
        if proc.returncode not in (0,-2,130):print(output[-2500:])
if __name__=='__main__':main()
