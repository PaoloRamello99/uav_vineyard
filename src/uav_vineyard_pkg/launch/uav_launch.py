from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='uav_vineyard_pkg',
            executable='map',
            name='map',
            output='screen'
        ),
        Node(
            package='uav_vineyard_pkg',
            executable='mission_control',
            name='mission_control',
            output='screen'
        ),
        Node(
            package='uav_vineyard_pkg',
            executable='navigation',
            name='navigation',
            output='screen'
        ),
        Node(
            package='uav_vineyard_pkg',
            executable='sprayer',
            name='sprayer',
            output='screen'
        ),
        Node(
            package='uav_vineyard_pkg',
            executable='refill',
            name='refill',
            output='screen'
        ),
        Node(
            package='uav_vineyard_pkg',
            executable='monitor',
            name='monitor',
            output='screen'
        ),
        Node(
            package='uav_vineyard_pkg',
            executable='dashboard',
            name='dashboard',
            output='screen'
        ),
    ])
