import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MissionControl(Node):
    def __init__(self):
        super().__init__('mission_control')
        self.cmd_pub = self.create_publisher(String, 'mission_cmd', 10)
        self.get_logger().info("Mission Control pronto. Scrivi 'start', 'stop' o 'restart'.")

        # Timer per leggere input da terminale
        self.timer = self.create_timer(1.0, self.user_input)

    def user_input(self):
        user_cmd = input("Inserisci comando (start/stop/restart): ").strip().lower()

        if user_cmd in ['start', 'stop', 'restart']:
            msg = String()
            msg.data = user_cmd.upper()  # diventa START, STOP o RESTART
            self.cmd_pub.publish(msg)
            self.get_logger().info(f"Missione: {msg.data}")
        else:
            self.get_logger().warn("Comando non valido (usa 'start', 'stop' o 'restart').")


def main(args=None):
    rclpy.init(args=args)
    node = MissionControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
