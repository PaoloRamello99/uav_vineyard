#!/usr/bin/env python3
############################################################################
# Offboard control per movimento a serpentina sopra filari di vigneto
# - movimento lungo X sopra il filare
# - quando raggiunge il fondo: spostamento laterale (Y) verso il filare successivo,
#   poi inversione della direzione X
# - semplice macchina a stati: "along" (lungo filare) e "shift" (traslazione laterale)
############################################################################

import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleStatus


class VineyardOffboard(Node):
    def __init__(self):
        super().__init__('vineyard_offboard')

        # QoS
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.status_sub = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile_sub)
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile_sub)

        # Publisher
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(
            TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)

        # Timer
        self.timer_period = 0.02  # seconds (50 Hz)
        self.timer = self.create_timer(self.timer_period, self.cmdloop_callback)
        self.dt = self.timer_period  # delta time comodo da usare

        # Parametri configurabili
        self.declare_parameter('row_length', 20.0)   # metri, lunghezza filare
        self.declare_parameter('row_spacing', 3.0)   # metri, distanza tra filari
        self.declare_parameter('num_rows', 4)        # numero filari da coprire
        self.declare_parameter('speed', 1.0)         # m/s lungo il filare
        self.declare_parameter('lateral_speed', 0.5) # m/s spostamento tra filari
        self.declare_parameter('altitude', 3.0)      # quota (positiva) sopra terreno

        self.row_length = float(self.get_parameter('row_length').value)
        self.row_spacing = float(self.get_parameter('row_spacing').value)
        self.num_rows = int(self.get_parameter('num_rows').value)
        self.speed = float(self.get_parameter('speed').value)
        self.lateral_speed = float(self.get_parameter('lateral_speed').value)
        self.altitude = float(self.get_parameter('altitude').value)

        # Stato interno
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # Posizione iniziale (sopra l'inizio del primo filare)
        self.x = 0.0
        self.y = 0.0
        self.direction = 1  # +1 = verso x crescente, -1 = verso x decrescente
        self.current_row = 0  # contatore: 0 = primo filare

        # Macchina a stati per movimento: 'along' (lungo filare) o 'shift' (spostamento laterale)
        self.state = 'along'
        self.shift_progress = 0.0  # metri spostati durante lo shift

        self.get_logger().info("Nodo vineyard_offboard avviato.")

    # Callback stato drone
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # Controllo offboard
    def cmdloop_callback(self):
        # Pubblica modalità offboard (richiesto continuamente)
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

        # Esegui solo se offboard + armato
        if not (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            return

        # Se missione completata -> log e non pubblica ulteriori setpoint (potresti volere un comportamento diverso)
        if self.current_row >= self.num_rows:
            self.get_logger().info("Missione completata: fine vigneto.")
            return

        # Stato: lungo filare
        if self.state == 'along':
            # Muovi lungo X con velocità definita (rispetta direzione)
            self.x += self.direction * self.speed * self.dt

            # Limiti estremi (clamp)
            if self.x > self.row_length:
                self.x = self.row_length
            if self.x < 0.0:
                self.x = 0.0

            # Controllo fine riga: se raggiungo l'estremo
            reached_end = (self.direction == 1 and self.x >= self.row_length - 1e-6) or \
                          (self.direction == -1 and self.x <= 0.0 + 1e-6)

            if reached_end:
                # Avvia fase di shift (spostamento laterale verso il filare successivo)
                self.state = 'shift'
                self.shift_progress = 0.0
                # NOTA: non incremento current_row qui — lo faccio al termine dello shift

        # Stato: spostamento laterale verso filare successivo
        elif self.state == 'shift':
            # Sposta y gradualmente (lateral_speed)
            step = self.lateral_speed * self.dt
            self.shift_progress += step
            # Progress bar: assegna alla y il valore (non cumulativo) per evitare errori numerici
            target_y = (self.current_row + 1) * self.row_spacing
            self.y = min(target_y, self.y + step)

            # Controllo completamento spostamento
            if abs(self.y - target_y) <= 1e-4 or self.shift_progress + 1e-9 >= self.row_spacing:
                # shift completato
                self.current_row += 1
                # inverti direzione sul nuovo filare
                self.direction *= -1
                self.state = 'along'
                # Mantieni x all'estremo prima dell'inversione (già impostato)
                # Se siamo arrivati oltre il numero di righe, verrà gestito all'inizio del prossimo ciclo

        # Prepara e pubblica il TrajectorySetpoint corrente
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        traj_msg.position[0] = float(self.x)
        traj_msg.position[1] = float(self.y)
        traj_msg.position[2] = -abs(float(self.altitude))  # PX4 usa NED: z negativo = sopra terra
        # opzionale: yaw orientato verso la direzione del movimento (0 = +x)
        traj_msg.yaw = 0.0 if self.direction == 1 else np.pi  # orientamento approssimato
        self.publisher_trajectory.publish(traj_msg)

        # Log sintetico per debug
        self.get_logger().info(
            f"Stato={self.state} | Fila {self.current_row+1}/{self.num_rows} | X={self.x:.2f} | Y={self.y:.2f} | dir={'→' if self.direction==1 else '←'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VineyardOffboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
