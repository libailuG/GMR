from pathlib import Path
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    share=Path(get_package_share_directory('human_fixed_hands_description'))
    urdf_path=share/'urdf/human.urdf'
    robot=ET.parse(urdf_path).getroot()
    # robot_description is XML text: resolve paths before its file context is lost.
    for mesh in robot.findall('.//mesh'):
        filename=mesh.get('filename', '')
        if filename and '://' not in filename and not Path(filename).is_absolute():
            mesh.set('filename', (urdf_path.parent/filename).resolve().as_uri())
    description=ET.tostring(robot, encoding='unicode')
    return LaunchDescription([
        DeclareLaunchArgument('gui',default_value='true'),
        DeclareLaunchArgument('rviz',default_value='true'),
        Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':description}],output='screen'),
        Node(package='joint_state_publisher_gui',executable='joint_state_publisher_gui',condition=IfCondition(LaunchConfiguration('gui'))),
        Node(package='joint_state_publisher',executable='joint_state_publisher',condition=UnlessCondition(LaunchConfiguration('gui'))),
        Node(package='rviz2',executable='rviz2',arguments=['-d',str(share/'rviz/human.rviz')],condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
