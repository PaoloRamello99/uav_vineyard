#!/usr/bin/env python3
############################################################################
# Offboard control per missione sopra vigneto:
# 1. Decollo dalla stazione di partenza
# 2. Avvicinamento al primo filare (movimento rettilineo con passo basato su distanza euclidea)
# 3. Movimento a serpentina lungo i filari per trattamenti
# 4. Ritorno alla stazione al termine della missione
############################################################################

import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Int32, String
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

        # Subscriber (stato veicolo)
        self.status_sub = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile_sub)
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile_sub)
        # Publisher (modalità + traiettoria)
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(
            TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)
    
        # Timer principale (50 Hz)
        self.timer_period = 0.02
        self.timer = self.create_timer(self.timer_period, self.cmdloop_callback)
        self.dt = self.timer_period

        # Parametri vigneto
        self.declare_parameter('home_x')
        self.declare_parameter('home_y')
        self.declare_parameter('home_z')
        self.declare_parameter('first_row_x')
        self.declare_parameter('first_row_y')
        self.declare_parameter('row_length')
        self.declare_parameter('row_spacing')
        self.declare_parameter('num_rows')
        self.declare_parameter('speed')
        self.declare_parameter('lateral_speed')
        self.declare_parameter('transit_speed')
        self.declare_parameter('altitude')

        #self.declare_parameter('home_x', 0.0)           # posizione X della stazione
        #self.declare_parameter('home_y', -5.0)          # posizione Y della stazione
        #self.declare_parameter('first_row_x', -10.0)    # posizione X del primo filare
        #self.declare_parameter('first_row_y', 20.0)     # posizione Y del primo filare
        #self.declare_parameter('row_length', 20.0)      # lunghezza filare (m)
        #self.declare_parameter('row_spacing', 2.5)      # distanza tra filari (m)
        #self.declare_parameter('num_rows', 10)          # numero di filari
        #self.declare_parameter('speed', 1.0)            # velocità lungo filare (m/s)
        #self.declare_parameter('lateral_speed', 0.5)    # velocità laterale (m/s)
        #self.declare_parameter('transit_speed', 2.0)    # velocità per il transito (m/s)
        #self.declare_parameter('altitude', 2.5)         # quota sopra terreno (m)

        # Lettura parametri
        self.home_x = self.get_parameter('home_x').get_parameter_value().double_value
        self.home_y = self.get_parameter('home_y').get_parameter_value().double_value
        self.home_z = self.get_parameter('home_z').get_parameter_value().double_value
        self.first_row_x = self.get_parameter('first_row_x').get_parameter_value().double_value
        self.first_row_y = self.get_parameter('first_row_y').get_parameter_value().double_value
        self.row_length = self.get_parameter('row_length').get_parameter_value().double_value
        self.row_spacing = self.get_parameter('row_spacing').get_parameter_value().double_value
        self.num_rows = self.get_parameter('num_rows').get_parameter_value().integer_value
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.lateral_speed = self.get_parameter('lateral_speed').get_parameter_value().double_value
        self.transit_speed = self.get_parameter('transit_speed').get_parameter_value().double_value
        self.altitude = self.get_parameter('altitude').get_parameter_value().double_value

        # Stato interno
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # Posizione iniziale (stazione)
        self.x = self.home_x
        self.y = self.home_y
        self.z = self.home_z
        
        # Macchina a stati
        self.state = 'takeoff'
        self.takeoff_timer = 0.0
        self.shift_progress = 0.0
        self.direction = 1 
        self.current_row = 0

        # Range filare
        self.row_x_min = min(self.first_row_x, self.first_row_x + self.row_length)
        self.row_x_max = max(self.first_row_x, self.first_row_x + self.row_length)

        self.get_logger().info("Nodo vineyard_offboard avviato: pronto per decollo.")

    # Callback per aggiornamento stato drone
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # Loop principale
    def cmdloop_callback(self):
        # Pubblica modalità offboard (deve essere pubblicata continuamente)
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        self.publisher_offboard_mode.publish(offboard_msg)

        # Verifica stato armato e offboard
        if not (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            return

        # Messaggio di traiettoria
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # ---------------------- MACCHINA A STATI ----------------------
        # 1) Decollo verticale
        if self.state == 'takeoff':
            traj_msg.position[0] = self.home_x
            traj_msg.position[1] = self.home_y
            traj_msg.position[2] = -self.altitude
            traj_msg.yaw = 0.0
            self.publisher_trajectory.publish(traj_msg)

            self.takeoff_timer += self.dt
            if self.takeoff_timer >= 3.0:  # attesa ~3s per stabilizzare
                # prepara il transito rettilineo
                self.state = 'to_first_row'
                self.get_logger().info("Decollo completato: Avvio transito al primo filare.")

        # 2) Spostamento verso il primo filare (movimento rettilineo con velocità costante)
        elif self.state == 'to_first_row':
            target_x = self.first_row_x
            target_y = self.first_row_y
            dx = target_x - self.x
            dy = target_y - self.y
            dist = np.hypot(dx, dy)
            # passo basato sulla velocità di transito
            step = self.transit_speed * self.dt

            if dist > 1e-6:
                if dist > step:
                    # movimento proporzionale al vettore (dx,dy)
                    self.x += (dx / dist) * step
                    self.y += (dy / dist) * step
                else:
                    # raggiunto il primo filare
                    self.x = target_x
                    self.y = target_y
                    # decidi direction in base alla posizione corrente rispetto agli estremi X del filare
                    if abs(self.x - self.row_x_min) < abs(self.x - self.row_x_max):
                        self.direction = 1
                    else:
                        self.direction = -1
                    self.state = 'along'
                    self.current_row = 1
                    self.get_logger().info("Raggiunto primo filare: inizio movimento lungo filare.")
                    
            traj_msg.position[0] = float(self.x)
            traj_msg.position[1] = float(self.y)
            traj_msg.position[2] = -float(self.altitude)
            traj_msg.yaw = 0.0
            self.publisher_trajectory.publish(traj_msg)

        # 3) Movimento lungo filare (serpentina)
        elif self.state == 'along':
            # avanzamento lungo X in base a direction e speed
            self.x += self.direction * self.speed * self.dt

            # clamp dentro l'intervallo del filare
            self.x = max(self.row_x_min, min(self.row_x_max, self.x))

            # controllo raggiungimento estremi del filare
            reached_end = (self.direction == 1 and self.x >= self.row_x_max - 1e-3) or \
                          (self.direction == -1 and self.x <= self.row_x_min + 1e-3)

            if reached_end:
                # Se completato l'ultimo filare → ritorna a casa
                if self.current_row >= self.num_rows:
                    self.state = 'return_home'
                    self.get_logger().info("Ultimo filare completato: rientro alla stazione.")
                else:
                # Altrimenti esegui lo shift al prossimo filare
                    self.state = 'shift'
                    self.shift_progress = 0.0 

            traj_msg.position[0] = float(self.x)
            traj_msg.position[1] = float(self.y)
            traj_msg.position[2] = -float(self.altitude)
            traj_msg.yaw = 0.0 if self.direction == 1 else np.pi
            self.publisher_trajectory.publish(traj_msg)

        # 4) Spostamento laterale tra filari (mantiene X e transla Y verso filare successivo)
        elif self.state == 'shift':
            # Sposta y gradualmente (lateral_speed)
            lateral_step = self.lateral_speed * self.dt
            self.shift_progress += lateral_step
            # Progress bar: assegna alla y il valore (non cumulativo) per evitare errori numerici
            target_y = self.first_row_y + (self.current_row) * self.row_spacing
            self.y = min(target_y, self.y + lateral_step)

            # Controllo completamento spostamento
            if abs(self.y - target_y) <= 1e-4 or self.shift_progress + 1e-9 >= self.row_spacing:
                # shift completato
                self.current_row += 1
                # inverti direzione sul nuovo filare
                self.direction *= -1
                self.state = 'along'
                # Mantieni x all'estremo prima dell'inversione (già impostato)

            traj_msg.position[0] = float(self.x)
            traj_msg.position[1] = float(self.y)
            traj_msg.position[2] = -float(self.altitude)
            traj_msg.yaw = 0.0 if self.direction == 1 else np.pi
            self.publisher_trajectory.publish(traj_msg)

        # 5) Ritorno alla stazione di partenza (strada più breve)
        elif self.state == 'return_home':
            target_x, target_y = self.home_x, self.home_y
            dx = target_x - self.x
            dy = target_y - self.y
            dist = np.hypot(dx, dy)
            step = self.transit_speed * self.dt

            if dist > 1e-6:
                if dist > step:
                    self.x += (dx / dist) * step
                    self.y += (dy / dist) * step
                else:
                    self.x, self.y = target_x, target_y
                    self.state = 'land'
                    self.get_logger().info("Rientrato alla stazione: avvio atterraggio.")

            traj_msg.position[0] = float(self.x)
            traj_msg.position[1] = float(self.y)
            traj_msg.position[2] = -float(self.altitude)
            traj_msg.yaw = np.pi
            self.publisher_trajectory.publish(traj_msg)

        # 6) Atterraggio
        elif self.state == 'land':
            traj_msg.position[0] = self.home_x
            traj_msg.position[1] = self.home_y
            traj_msg.position[2] = self.home_z  # scende a terra
            traj_msg.yaw = 0.0
            self.publisher_trajectory.publish(traj_msg)
            self.dz = abs(self.z - self.home_z)

            if self.dz < 1e-6:
                self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
                self.get_logger().info("Atterraggio completato. Missione finita.")

        # Log per monitoraggio
        self.get_logger().info(
            f"Stato={self.state} | Fila={self.current_row}/{self.num_rows} | X={self.x:.2f} | Y={self.y:.2f} "
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
