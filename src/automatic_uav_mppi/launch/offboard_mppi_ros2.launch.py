from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():

    # --- EXECUTABLE NODES ---
    uav_offboard_mppi = Node(
        package='automatic_uav_mppi',
        executable='uav_offboard_mppi',
        name='uav_offboard_mppi',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': True}],
        remappings=[
            ("fmu/out/vehicle_status", "fmu/out/vehicle_status_v1"),
            ("fmu/out/vehicle_local_position", "fmu/out/vehicle_local_position_v1"),
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


    return LaunchDescription([
        LogInfo(msg="🚁 Starting UAV Offboard MPPI Control..."),
        uav_offboard_mppi,
        serpentine_trajectory
    ])