import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import BatteryStatus
from std_msgs.msg import Float32

class BatteryNode(Node):
    def __init__(self):
        super().__init__('battery_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            BatteryStatus,
            '/fmu/out/battery_status_v1',
            self.listener_callback,
            qos_profile
        )
        self.publisher = self.create_publisher(Float32, '/uav/battery_state', 10)
        self.get_logger().info("BatteryNode attivo — in ascolto su /fmu/out/battery_status_v1")

    def listener_callback(self, msg):
        battery_pct = msg.remaining * 100.0
        self.get_logger().info(f"Batteria {battery_pct:.1f}% (warning {msg.warning})")

        msg_out = Float32()
        msg_out.data = battery_pct
        self.publisher.publish(msg_out)

def main():
    rclpy.init()
    node = BatteryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
