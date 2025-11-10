import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class Map(Node):
    def __init__(self):
        super().__init__('map')
        self.create_subscription(String, 'status_refill', self.refill_callback, 10)
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(Int32, 'battery_level', self.battery_callback, 10)
        self.create_subscription(Int32, 'pesticide_level', self.pesticide_callback, 10)

        self.tot_filari = 10
        self.completed_filari = 0
        self.refill_count = 0
        self.mission_active = False
        self.last_battery = 100
        self.last_pesticide = 100

    def cmd_callback(self, msg):
        cmd = msg.data.upper()
        if cmd == "START":
            self.mission_active = True
            self.completed_filari = 0
            self.refill_count = 0
            self.get_logger().info("Missione avviata.")
        elif cmd == "STOP":
            self.mission_active = False
            self.get_logger().info("Missione interrotta.")
        elif cmd == "RESTART":
            self.mission_active = True
            self.get_logger().info("Missione ripresa.")

    def refill_callback(self, msg):
        if msg.data == "REFILL_DONE":
            self.refill_count += 1
            self.get_logger().info(f"Refill numero {self.refill_count} completato.")

    def battery_callback(self, msg):
        self.last_battery = msg.data

    def pesticide_callback(self, msg):
        self.last_pesticide = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = Map()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
