import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Sprayer(Node):
    def __init__(self):
        super().__init__('sprayer')

        # Sub
        self.create_subscription(String, 'position', self.position_callback, 10)
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)

        # Pub
        self.sprayer_pub = self.create_publisher(String, 'sprayer_cmd', 10)

        self.active = False

    def cmd_callback(self, msg):
        if msg.data == "STOP":
            self.active = False
            self.send_cmd("OFF")
            self.get_logger().info("Pompa spenta (missione interrotta).")

    def position_callback(self, msg):
        pos = msg.data

        if pos.endswith("_PALO_1"):   # esattamente il primo palo
            self.active = True
            self.send_cmd("ON")
            self.get_logger().info(f"Pompa ACCESA in {pos}")

        elif pos.endswith("_PALO_50"):  # esattamente l’ultimo palo
            self.active = False
            self.send_cmd("OFF")
            self.get_logger().info(f"Pompa SPENTA in {pos}")

        elif pos == "STAZIONE":  # ritorno in stazione
            self.active = False
            self.send_cmd("OFF")
            self.get_logger().info("Pompa SPENTA (stazione).")

    def send_cmd(self, state):
        msg = String()
        msg.data = state
        self.sprayer_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Sprayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
