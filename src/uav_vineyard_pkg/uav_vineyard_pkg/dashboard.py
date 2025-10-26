import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

class Dashboard(Node):
    def __init__(self):
        super().__init__('dashboard')

        # Subscriptions
        self.create_subscription(String, 'mission_cmd', self.cmd_callback, 10)
        self.create_subscription(String, 'status_refill', self.refill_callback, 10)
        self.create_subscription(String, 'position', self.position_callback, 10)
        self.create_subscription(Int32, 'battery_level', self.battery_callback, 10)
        self.create_subscription(Int32, 'pesticide_level', self.pesticide_callback, 10)
        self.create_subscription(String, 'sprayer_cmd', self.sprayer_callback, 10)

        # Stato missione
        self.tot_filari = 10
        self.completed_filari = 0
        self.refill_count = 0
        self.mission_active = False

        # Risorse
        self.battery = 100
        self.pesticide = 100
        self.pump = "OFF"

        # Live dashboard
        self.live = Live(self.generate_table(), console=console, refresh_per_second=4, screen=True)
        self.live.start()

        # Timer per aggiornare dashboard
        self.create_timer(0.5, self.update_dashboard)  # aggiorna due volte al secondo

    # ---------------- Callbacks ---------------- #
    def cmd_callback(self, msg):
        if msg.data == "START":
            self.mission_active = True
            self.completed_filari = 0
            self.refill_count = 0
        elif msg.data == "STOP":
            self.mission_active = False
        elif msg.data == "RESTART":
            self.mission_active = True

    def refill_callback(self, msg):
        if msg.data == "REFILL_DONE":
            self.refill_count += 1

    def position_callback(self, msg):
        if msg.data.endswith("PALO_50") and self.completed_filari < self.tot_filari:
            self.completed_filari += 1
        elif msg.data == "FINISH":
            self.mission_active = False

    def battery_callback(self, msg):
        self.battery = msg.data

    def pesticide_callback(self, msg):
        self.pesticide = msg.data

    def sprayer_callback(self, msg):
        self.pump = msg.data

    # ---------------- Dashboard ---------------- #
    def generate_table(self):
        table = Table(title="DRONE DASHBOARD - LIVE", title_style="bold red")
        table.add_column("Elemento", style="cyan", no_wrap=True)
        table.add_column("Valore", style="magenta")

        table.add_row("Filari completati", f"{self.completed_filari} / {self.tot_filari}")
        table.add_row("Refill eseguiti", str(self.refill_count))
        table.add_row("Batteria", f"{self.battery} %")
        table.add_row("Pesticida", f"{self.pesticide} %")
        table.add_row("Pompa", self.pump)
        table.add_row("Missione", "ATTIVA" if self.mission_active else "FERMA")
        return table

    def update_dashboard(self):
        self.live.update(self.generate_table())

    def destroy_node(self):
        self.live.stop()
        super().destroy_node()

# ---------------- Main ---------------- #
def main(args=None):
    rclpy.init(args=args)
    node = Dashboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
