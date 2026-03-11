import shutil
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # --- 0. PRE-LAUNCH SETUP ---
    source_world = '/workspaces/uav_vineyard/src/automatic_uav_mppi/worlds/vineyard_world.sdf'
    target_world_dir = '/workspaces/PX4-Autopilot/Tools/simulation/gz/worlds/'
    
    if os.path.exists(source_world):
        shutil.copy(source_world, target_world_dir)
        print(f"✅ World file successfully copied to: {target_world_dir}")
    else:
        print(f"❌ WARNING: Original world file not found at: {source_world}")

    # --- 1. ARGUMENTS ---
    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value='/workspaces/PX4-Autopilot',
        description='Path to the PX4 Firmware root directory'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='vineyard_world',
        description='Gazebo world to load'
    )

    # --- 2. INFRASTRUCTURE NODES ---
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=['/world/vineyard_world/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        remappings=[('/world/vineyard_world/clock', '/clock')],
        output='screen'
    )

    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
        name='micro_xrce_agent'
    )

    px4_sitl = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500'],
        cwd=LaunchConfiguration('px4_dir'),
        output='screen',
        additional_env={
            'PX4_SIM_SPEED_FACTOR': '0.5',  # <--- Real Time Factor a 0.5
            'PX4_GZ_WORLD': LaunchConfiguration('world') 
        },
        name='px4_sitl'
    )

    return LaunchDescription([
        LogInfo(msg="🚀 Initializing PX4 and Gazebo environment..."),
        px4_dir_arg,
        world_arg,
        ros_gz_bridge,
        micro_xrce_agent,
        px4_sitl
    ])