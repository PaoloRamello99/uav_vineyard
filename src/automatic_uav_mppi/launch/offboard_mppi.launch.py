from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # --- 1. ARGUMENTS ---
    px4_dir_arg = DeclareLaunchArgument(
        "px4_dir",
        default_value="/home/riccardo/px4/PX4-Autopilot",  # FIXME: Cambiare il path di px4
        description="Path to the PX4 Firmware root directory",
    )

    # world_arg = DeclareLaunchArgument(
    #     "world", default_value="vineyard_world", description="Gazebo world to load"
    # )

    # --- 2. INFRASTRUCTURE NODES ---
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    micro_xrce_agent = ExecuteProcess(
        cmd=["MicroXRCEAgent", "udp4", "-p", "8888"],
        output="screen",
        name="micro_xrce_agent",
    )

    px4_sitl = ExecuteProcess(
        cmd=["make", "px4_sitl", "gz_x500"],
        cwd=LaunchConfiguration("px4_dir"),
        output="screen",
        # additional_env={"PX4_GZ_WORLD": LaunchConfiguration("world")},
        name="px4_sitl",
    )

    # --- 3. EXECUTABLE NODES ---
    mppi_rate_node = Node(
        package="automatic_uav_mppi",
        executable="mppi_rate_node",
        name="mppi_rate_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"use_sim_time": False}],
        remappings=[
            ("fmu/out/vehicle_status", "fmu/out/vehicle_status_v1"),
            ("fmu/out/vehicle_local_position", "fmu/out/vehicle_local_position_v1"),
        ],
    )

    reference_node = Node(
        package="automatic_uav_mppi",
        executable="reference_node",
        name="reference_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"use_sim_time": False}],
    )

    # --- 4. CONTROL FUNCTION ---
    def on_px4_log_check(event):
        """
        Function to check PX4 logs and start MPPI and reference nodes on takeoff message.
        """
        if "Ready for takeoff!" in event.text.decode():
            return [
                LogInfo(msg="Starting MPPI and reference nodes..."),
                mppi_rate_node,
                reference_node,
            ]
        return None

    spawn_mppi_on_takeoff = RegisterEventHandler(
        event_handler=OnProcessIO(target_action=px4_sitl, on_stdout=on_px4_log_check)
    )

    return LaunchDescription(
        [
            LogInfo(msg="🚀 Initializing environment..."),
            px4_dir_arg,
            # world_arg,
            ros_gz_bridge,
            micro_xrce_agent,
            px4_sitl,
            spawn_mppi_on_takeoff,
        ]
    )
