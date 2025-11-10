import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import String

class Sprayer(Node):
    def __init__(self):
        super().__init__('sprayer')
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, 10)
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.sprayer_pub = self.create_publisher(String, 'sprayer_cmd', 10)
        self.row_length = 80.0
        self.pump_active = False
        self.get_logger().info("Sprayer attivo.")

    def cmd_callback(self, msg):
        if msg.data == "STOP":
            self.send_cmd("OFF")

    def pos_callback(self, msg):
        x = msg.x
        if not self.pump_active and abs(x - 0.0) < 0.5:
            self.send_cmd("ON")
            self.pump_active = True
            self.get_logger().info(f"Pompa ACCESA (X={x:.1f})")
        elif self.pump_active and abs(x - self.row_length) < 0.5:
            self.send_cmd("OFF")
            self.pump_active = False
            self.get_logger().info(f"Pompa SPENTA (X={x:.1f})")

    def send_cmd(self, state):
        self.sprayer_pub.publish(String(data=state))

def main(args=None):
    rclpy.init(args=args)
    node = Sprayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
