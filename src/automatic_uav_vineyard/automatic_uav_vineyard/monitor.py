#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String


class Monitor(Node):
    def __init__(self):
        super().__init__('monitor')
        self.declare_parameter('low_battery', 20)
        self.threshold = int(self.get_parameter('low_battery').value)
        self.last_warning = None

        self.create_subscription(Int32, 'battery_level', self.battery_callback, 10)
        self.warning_pub = self.create_publisher(String, 'battery_warning', 10)

        self.get_logger().info("Nodo 'monitor' avviato (soglia batteria = %d%%)." % self.threshold)

    def battery_callback(self, msg: Int32):
        level = msg.data
        if level <= self.threshold:
            if self.last_warning != "LOW_BATTERY":
                warning = String()
                warning.data = "LOW_BATTERY"
                self.warning_pub.publish(warning)
                self.last_warning = "LOW_BATTERY"
                self.get_logger().warn(f"Batteria sotto soglia ({level}%). Warning pubblicato.")
        else:
            self.last_warning = None


def main(args=None):
    rclpy.init(args=args)
    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
