import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class PesticideNode(Node):
    def __init__(self):
        super().__init__('pesticide_node')

        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(String, 'status_refill', self.refill_callback, 10)

        self.pesticide_pub = self.create_publisher(Int32, 'pesticide_level', 10)

        self.pesticide = 100
        self.mission_active = False
        self.timer_pesticide = self.create_timer(3.0, self.check_pesticide)

    def cmd_callback(self, msg):
        if msg.data == "START":         # Avvio missione
            self.mission_active = True
            self.pesticide = 100
            self.get_logger().info("Missione avviata da capo. Pesticida ok.")

        elif msg.data == "STOP":        # Stop missione e ritorno alla base
            self.mission_active = False
            self.pesticide = 100
            self.get_logger().info("Missione interrotta: drone in stazione e ricaricato al 100%.")

        elif msg.data == "RESTART":     # Riprendi missione dal punto di stop
            self.mission_active = True
            self.pesticide = 100
            self.get_logger().info("Missione ripresa dal punto di stop, con pesticida ricaricato.")


    def refill_callback(self, msg):
        if msg.data == "REFILL_DONE":
            self.pesticide = 100
            self.get_logger().info("Pesticida ricaricato.")
            self.mission_active = True

    def finish_mission(self, msg):
        if msg.data == "FINISH":
            self.get_logger().info("FINE MISSIONE")
            self.get_logger().info(f"Pesticida rimasto: {self.pesticide}%")
            self.mission_active = False

    def check_pesticide(self):
        if self.mission_active:
            self.pesticide -= 5
            self.pesticide_pub.publish(Int32(data=self.pesticide))
            self.get_logger().info(f"Pesticide: {self.pesticide}%")

def main(args=None):
    rclpy.init(args=args)
    node = PesticideNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
