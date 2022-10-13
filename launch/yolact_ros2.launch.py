from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from launch import LaunchDescription

params_file = 'yolact_config.yaml'

def generate_launch_description():

    #Load params

    pkg_dir = get_package_share_directory('yolact_ros2')
    config_file = os.path.join(pkg_dir, 'config', params_file)

    stdout_stdout_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_USE_STDOUT', '1')
    
    stdout_bufstream_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    #Create Node:

    yolact_ros_node = Node(
    package='yolact_ros2',
    executable='yolact_ros2_node',
    name='yolact_ros2_node',
    output='screen',
    parameters=[config_file]
    )


    ld = LaunchDescription()

    ld.add_action(stdout_stdout_envvar)
    ld.add_action(stdout_bufstream_envvar)
    ld.add_action(yolact_ros_node)

    return ld
