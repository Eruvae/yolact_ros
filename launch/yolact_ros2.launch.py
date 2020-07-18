from ament_index_python.packages import get_package_share_directory
import os
from launch import LaunchDescription
from launch_ros.actions import Node

params_file = 'yolact_config.yaml'

def generate_launch_description():

    #Load params
    pkg_dir = get_package_share_directory('yolact_ros2')
    config_file = os.path.join(pkg_dir, 'config', params_file)

    #Create Node:
    yolact_ros_node = Node(
    package='yolact_ros2',
    node_executable='yolact_ros2_node',
    node_name='yolact_ros2_node',
    output='screen',
    parameters=[config_file]
    )


    ld = LaunchDescription()

    ld.add_action(yolact_ros_node)

    return ld
