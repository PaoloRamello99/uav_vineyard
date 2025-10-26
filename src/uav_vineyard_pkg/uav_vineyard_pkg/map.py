import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

class Map(Node):
    def __init__(self):
        super().__init__('map')

        # Sub
        self.create_subscription(String, 'position', self.position_callback, 10)
        self.create_subscription(String, 'status_refill', self.refill_callback, 10)
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(Int32, 'battery_level', self.battery_callback, 10)
        self.create_subscription(Int32, 'pesticide_level', self.pesticide_callback, 10)

        # Stato missione
        self.tot_filari = 10
        self.completed_filari = 0
        self.refill_count = 0
        self.mission_active = False

        # Stato risorse
        self.last_battery = 100
        self.last_pesticide = 100

    def cmd_callback(self, msg):
        if msg.data == "START":
            self.mission_active = True
            self.completed_filari = 0
            self.refill_count = 0
            self.get_logger().info("Map: missione iniziata, azzerati i contatori.")
        elif msg.data == "STOP":
            self.mission_active = False
            self.get_logger().info("Map: missione interrotta.")
        elif msg.data == "RESTART":
            self.mission_active = True
            self.get_logger().info("Map: missione ripresa.")

    def position_callback(self, msg):
        if not self.mission_active:
            return

        pos = msg.data
        if pos.endswith("PALO_50"):  # fine filare preciso
            self.completed_filari += 1
            remaining = self.tot_filari - self.completed_filari
            self.get_logger().info(f"Filare {self.completed_filari} completato. Filari restanti: {remaining}")

        elif pos == "STAZIONE":
            self.get_logger().info("Drone in stazione per ricarica.")
    
        elif pos == "FINISH":
            self.get_logger().info("Drone in stazione. Missione terminata.")
            self.print_report()
            self.mission_active = False

    def refill_callback(self, msg):
        if msg.data == "REFILL_DONE":
            self.refill_count += 1
            self.get_logger().info(f"Refill numero {self.refill_count} completato.")

    def battery_callback(self, msg):
        self.last_battery = msg.data

    def pesticide_callback(self, msg):
        self.last_pesticide = msg.data

    def print_report(self):
        self.get_logger().info("===== REPORT FINALE MISSIONE =====")
        self.get_logger().info(f"Filari completati: {self.completed_filari}/{self.tot_filari}")
        self.get_logger().info(f"Refill effettuati: {self.refill_count}")
        self.get_logger().info(f"Batteria residua: {self.last_battery}%")
        self.get_logger().info(f"Pesticida residuo: {self.last_pesticide}%")
        self.get_logger().info("=================================")

def main(args=None):
    rclpy.init(args=args)
    node = Map()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
