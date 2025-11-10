#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from px4_msgs.msg import BatteryStatus
import time


class RefillNode(Node):
    def __init__(self):
        super().__init__('refill_node')

        self.create_subscription(String, 'position', self.position_callback, 10)
        self.create_subscription(Int32, 'battery_level', self.battery_callback, 10)
        self.px4_battery_pub = self.create_publisher(BatteryStatus, '/fmu/in/battery_status', 10)
        self.status_pub = self.create_publisher(String, 'status_refill', 10)

        self.position = None
        self.battery_level = 100
        self.refilling = False

        self.get_logger().info("Nodo 'refill_node' avviato.")

    def position_callback(self, msg: String):
        self.position = msg.data
        self.check_and_refill()

    def battery_callback(self, msg: Int32):
        self.battery_level = msg.data
        self.check_and_refill()

    def check_and_refill(self):
        if self.position == "STAZIONE" and self.battery_level <= 20 and not self.refilling:
            self.refilling = True
            self.get_logger().info("Drone in stazione, batteria bassa — inizio ricarica.")
            self.do_refill()

    def do_refill(self):
        for i in range(20, 101, 10):
            px4_msg = BatteryStatus()
            px4_msg.remaining = i / 100.0
            self.px4_battery_pub.publish(px4_msg)
            self.get_logger().info(f"Ricarica: {i}%")
            time.sleep(0.5)

        msg = String()
        msg.data = "REFILL_DONE"
        self.status_pub.publish(msg)
        self.get_logger().info("Ricarica completata, REFILL_DONE pubblicato.")
        self.refilling = False


def main(args=None):
    rclpy.init(args=args)
    node = RefillNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
