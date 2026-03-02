import os
import shutil
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    home_dir = os.path.expanduser('~')
    
    # --- 1. ARGUMENTS ---
    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value=os.path.join(home_dir, 'PX4-Autopilot'),
        description='Path to the PX4 Firmware root directory'
    )

    # --- 2. INFRASTRUCTURE NODES ---
    
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
            'PX4_SIM_SPEED_FACTOR': '1.0',  
        },
        name='px4_sitl'
    )

    # --- 3. EXECUTABLE NODES ---
    uav_offboard_mppi = Node(
        package='automatic_uav_mppi',
        executable='uav_offboard_mppi',
        name='uav_offboard_mppi',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': False}], 
        remappings=[
            ("/fmu/out/vehicle_status", "/fmu/out/vehicle_status_v1"),
            ("/fmu/out/vehicle_local_position", "/fmu/out/vehicle_local_position_v1"),
        ],
    )

    serpentine_trajectory = Node(
        package="automatic_uav_mppi",
        executable="serpentine_trajectory",
        name="serpentine_trajectory",
        output="screen",
        emulate_tty=True,
        parameters=[{"use_sim_time": False}], 
    )

    # --- 4. CONTROL FUNCTION ---
    def on_px4_log_check(event):
        """
        Function to check PX4 logs and start MPPI node on takeoff message.
        """
        if 'Ready for takeoff!' in event.text.decode():
            return [
                LogInfo(msg="Starting MPPI node..."),
                uav_offboard_mppi,
                serpentine_trajectory  
            ]
        return None

    spawn_mppi_on_takeoff = RegisterEventHandler(
        event_handler=OnProcessIO(
            target_action=px4_sitl,
            on_stdout=on_px4_log_check
        )
    )

    return LaunchDescription([
        LogInfo(msg="🚀 Initializing environment (Real Time, No Sim Clock)..."),
        px4_dir_arg,
        micro_xrce_agent,
        px4_sitl,
        spawn_mppi_on_takeoff
    ])