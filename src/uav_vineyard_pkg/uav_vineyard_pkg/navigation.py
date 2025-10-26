import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')

        # Sub
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(String, 'status_warning', self.warning_callback, 10)
        self.create_subscription(String, 'status_refill', self.refill_callback, 10)

        # Pub
        self.position_pub = self.create_publisher(String, 'position', 10)

        # Stato missione
        self.mission_active = False
        self.warning_received = False
        self.current_filare = 1
        self.current_palo = 0

        # Parametri campo
        self.numero_filari = 10
        self.numero_pali_filare = 50

        # Timer simulazione movimento
        self.timer = self.create_timer(0.3, self.publish_position)

    # --- CALLBACKS ---
    def cmd_callback(self, msg):
        if msg.data == "START":
            self.mission_active = True
            self.warning_received = False
            self.current_filare = 1
            self.current_palo = 0
            self.get_logger().info("Missione avviata.")
        elif msg.data == "STOP":
            self.mission_active = False
            self.get_logger().info("Missione interrotta manualmente.")
        elif msg.data == "RESTART":
            self.mission_active = True
            self.get_logger().info(f"Missione ripresa da FILARE {self.current_filare}, PALO {self.current_palo}.")

    def warning_callback(self, msg):
        self.warning_received = True
        self.get_logger().warn(f"Warning ricevuto: {msg.data}")

    def refill_callback(self, msg):
        if msg.data == "REFILL_DONE":
            self.get_logger().info("Refill completato, missione riprende.")
            self.mission_active = True
            self.warning_received = False

    # --- LOGICA DI MOVIMENTO ---
    def publish_position(self):
        if not self.mission_active:
            return

        # Se warning ricevuto → torna in stazione dopo questo palo
        if self.warning_received:
            self.get_logger().warn("Risorsa bassa: torno in stazione")
            self.position_pub.publish(String(data="STAZIONE"))
            self.get_logger().info("Drone in STAZIONE (in attesa refill).")
            self.mission_active = False
            return

        # Avanzamento palo
        self.current_palo += 1

        # Se finito il filare → passa al successivo
        if self.current_palo > self.numero_pali_filare:
            self.current_palo = 1
            self.current_filare += 1

            if self.current_filare > self.numero_filari:
                self.get_logger().info("Missione completata! Ritorno in stazione.")
                self.position_pub.publish(String(data="FINISH"))
                self.position_pub.publish(String(data="STAZIONE"))
                self.mission_active = False
                return

        # Pubblica posizione corrente
        pos_msg = f"FILARE_{self.current_filare}_PALO_{self.current_palo}"
        self.position_pub.publish(String(data=pos_msg))
        self.get_logger().info(f"Drone in {pos_msg}")


def main(args=None):
    rclpy.init(args=args)
    node = Navigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
