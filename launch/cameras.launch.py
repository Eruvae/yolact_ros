#!/usr/bin/env python3
from launch import LaunchDescription
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

import os
def generate_launch_description():
    namespace   = LaunchConfiguration('namespace',      default='')
    pkg_dir = get_package_share_directory('yolact_ros2')
    params_file = os.path.join(pkg_dir, 'config', 'cameras.yaml')

    remappings = [
      ('/tf',					'tf'),
      ('/tf_static',  'tf_static')]

    param_substitutions = {'namespace': namespace}

    configure_params = RewrittenYaml(
            source_file	    = params_file,
            root_key        = namespace,
            param_rewrites  = param_substitutions,
            convert_types   = True
        )

    ld = LaunchDescription()

    return LaunchDescription([
        Node(
                package='realsense2_camera',
                name='camera_node_right',
                namespace=namespace,
                executable='realsense2_camera_node',
                parameters=[configure_params],
                remappings = remappings,
                output='screen',
                emulate_tty=True,
            )
    ])
