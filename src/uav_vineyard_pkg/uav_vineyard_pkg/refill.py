import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class Refill(Node):
    def __init__(self):
        super().__init__('refill')

        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(String, 'position', self.position_callback, 10)
        self.create_subscription(Int32, 'battery_level', self.batterylevel_callback, 10)
        self.create_subscription(Int32, 'pesticide_level', self.pesticidelevel_callback, 10)

        self.refill_pub = self.create_publisher(String, 'status_refill', 10)

        self.active = False
        self.battery_warning = 20
        self.pesticide_warning = 20
        self.low_resource = False

    def cmd_callback(self, msg):
        if msg.data == "START":
            self.active = True
        elif msg.data == "STOP":
            self.active = False
        elif msg.data == "RESTART":
            self.active = True

    def batterylevel_callback(self, msg):
        if msg.data <= self.battery_warning:
            self.low_resource = True

    def pesticidelevel_callback(self, msg):
        if msg.data <= self.pesticide_warning:
            self.low_resource = True

    def position_callback(self, msg):
        if msg.data == "STAZIONE" and self.low_resource:
            self.get_logger().info("Drone in stazione -> Ricarica in corso...")
            self.do_refill()
        
        elif msg.data == "FINISH":
            self.get_logger().info("Drone in stazione per fine missione")

    def do_refill(self):
        msg = String()
        msg.data = "REFILL_DONE"
        self.refill_pub.publish(msg)
        self.low_resource = False
        self.get_logger().info("Refill completato!")


def main(args=None):
    rclpy.init(args=args)
    node = Refill()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
