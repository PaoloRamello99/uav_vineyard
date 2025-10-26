import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class Monitor(Node):
    def __init__(self):
        super().__init__('monitor')

        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(String, 'status_refill', self.refill_callback, 10)
        self.create_subscription(String, 'position', self.finish_mission, 10)

        self.warning_pub = self.create_publisher(String, 'status_warning', 10)
        self.battery_pub = self.create_publisher(Int32, 'battery_level', 10)
        self.pesticide_pub = self.create_publisher(Int32, 'pesticide_level', 10)

        self.battery = 100
        self.pesticide = 100
        self.mission_active = False
        self.timer_battery = self.create_timer(3.0, self.check_battery)
        self.timer_pesticide = self.create_timer(4.0, self.check_pesticide)

    def cmd_callback(self, msg):
        if msg.data == "START":
            # Nuova missione da capo
            self.mission_active = True
            self.battery = 100
            self.pesticide = 100
            self.get_logger().info("Missione avviata da capo. Batterie e serbatoio pieni.")

        elif msg.data == "STOP":
            # Stop missione e refill automatico
            self.mission_active = False
            self.battery = 100
            self.pesticide = 100
            self.get_logger().info("Missione interrotta: drone in stazione e ricaricato al 100%.")

        elif msg.data == "RESTART":
            # Ripartenza dal punto fermato, ma con valori ricaricati
            self.mission_active = True
            self.battery = 100
            self.pesticide = 100
            self.get_logger().info("Missione ripresa dal punto di stop, con batteria e pesticida ricaricati.")


    def refill_callback(self, msg):
        if msg.data == "REFILL_DONE":
            self.battery = 100
            self.pesticide = 100
            self.get_logger().info("Monitor: batteria e pesticida ricaricati.")
            self.mission_active = True

    def finish_mission(self, msg):
        if msg.data == "FINISH":
            self.get_logger().info("FINE MISSIONE")
            self.get_logger().info(f"Batteria rimasta: {self.battery}%")
            self.get_logger().info(f"Pesticida rimasto: {self.pesticide}%")
            self.mission_active = False

    def check_battery(self):
        if self.mission_active:
            self.battery -= 5
            self.battery_pub.publish(Int32(data=self.battery))
            self.get_logger().info(f"Battery: {self.battery}%")
            if self.battery <= 20:
                self.warning_pub.publish(String(data="LOW_BATTERY"))
                self.get_logger().warn("Batteria bassa! Mandato segnale a Navigation.")

    def check_pesticide(self):
        if self.mission_active:
            self.pesticide -= 7
            self.pesticide_pub.publish(Int32(data=self.pesticide))
            self.get_logger().info(f"Pesticida: {self.pesticide}%")
            if self.pesticide <= 20:
                self.warning_pub.publish(String(data="LOW_PESTICIDE"))
                self.get_logger().warn("Pesticida basso! Mandato segnale a Navigation.")

def main(args=None):
    rclpy.init(args=args)
    node = Monitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
