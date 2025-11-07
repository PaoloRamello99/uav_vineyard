from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('automatic_uav_vineyard'),
        'config',
        'vineyard_params.yaml'
    )

    return LaunchDescription([

        Node(
            package='automatic_uav_vineyard',
            executable='vineyard_offboard',
            name='vineyard_offboard',
            output='screen',
            parameters=[config]
        ),

        Node(
            package='automatic_uav_vineyard',
            executable='refill_node',
            name='refill_node',
            output='screen',
            parameters=[config]
        ),

        Node(
            package='automatic_uav_vineyard',
            executable='pesticide_node',
            name='pesticide_node',
            output='screen',
            parameters=[config]
        ),

        Node(
            package='automatic_uav_vineyard',
            executable='monitor_node',
            name='monitor_node',
            output='screen',
            parameters=[config]
        )
    ])
