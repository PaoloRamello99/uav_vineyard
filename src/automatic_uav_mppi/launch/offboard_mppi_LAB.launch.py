import os
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
    
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
        parameters=[{'use_sim_time': True}]
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
            'PX4_SIM_SPEED_FACTOR': '0.3',  
            'GZ_VERSION': 'harmonic' 
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
        parameters=[{'use_sim_time': True}], 
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
        parameters=[{"use_sim_time": True}], 
    )

    # --- 4. CONTROL FUNCTION ---
    def on_px4_log_check(event):
        """
        Start the MPPI nodes only when PX4 is fully initialized and ready.
        """
        if 'Ready for takeoff!' in event.text.decode():
            return [
                LogInfo(msg="🚀 Clock synchronized. Launching MPPI and Trajectory..."),
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
        LogInfo(msg="Drone Lab: Gazebo Harmonic synchronization active..."),
        px4_dir_arg,
        ros_gz_bridge, 
        micro_xrce_agent,
        px4_sitl,
        spawn_mppi_on_takeoff
    ])