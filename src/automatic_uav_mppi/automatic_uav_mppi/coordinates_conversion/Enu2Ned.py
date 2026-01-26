import numpy as np

class Enu2NedConverter:
    """
    Converte lo stato da ENU (East-North-Up) a NED (North-East-Down).
    Standard: ROS -> PX4
    """
    
    # Matrice di permutazione: Scambia X/Y e inverte Z
    R_matrix = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, -1]])

    @staticmethod
    def enu_to_ned(x_enu):
        """
        Input x_enu: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r] (ENU / FLU)
        Output x_ned: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r] (NED / FRD)
        """
        x_ned = np.zeros_like(x_enu)

        # ---------------------------------------------------------
        # 1. Posizione e Velocità
        # ---------------------------------------------------------
        x_ned[0:3] = Enu2NedConverter.R_matrix @ x_enu[0:3]
        x_ned[3:6] = Enu2NedConverter.R_matrix @ x_enu[3:6]

        # ---------------------------------------------------------
        # 2. Quaternione
        # ---------------------------------------------------------
        # Trasformazione analitica diretta:
        # Corregge l'offset di Yaw di +90 gradi (Nord ENU -> Nord NED)
        qw, qx, qy, qz = x_enu[6:10]
        c = 0.70710678118  # 1 / sqrt(2)

        x_ned[6] = c * (qw + qz)      # w
        x_ned[7] = c * (qx + qy)      # x
        x_ned[8] = c * (qx - qy)      # y
        x_ned[9] = c * (qw - qz)      # z

        # ---------------------------------------------------------
        # 3. Body Rates
        # ---------------------------------------------------------
        # FLU (Forward-Left-Up) -> FRD (Forward-Right-Down)
        
        p_flu, q_flu, r_flu = x_enu[10:13]
        
        x_ned[10] = p_flu   # p (Roll)
        x_ned[11] = -q_flu  # q (Pitch)
        x_ned[12] = -r_flu  # r (Yaw)

        return x_ned