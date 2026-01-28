import numpy as np

class Ned2EnuConverter:
    """
    Converte lo stato da NED (North-East-Down) a ENU (East-North-Up).
    Standard: PX4 -> ROS
    """
    
    # Matrice di permutazione per Posizione e Velocità Lineare
    # ENU_x (East)  = NED_y
    # ENU_y (North) = NED_x
    # ENU_z (Up)    = -NED_z
    R_matrix = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, -1]])

    @staticmethod
    def ned_to_enu(x_ned):
        """
        Input x_ned: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r] (NED / FRD)
        Output x_enu: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r] (ENU / FLU)
        """
        x_enu = np.zeros_like(x_ned)

        # ---------------------------------------------------------
        # 1. Posizione e Velocità (Vettori)
        # ---------------------------------------------------------
        x_enu[0:3] = Ned2EnuConverter.R_matrix @ x_ned[0:3]
        x_enu[3:6] = Ned2EnuConverter.R_matrix @ x_ned[3:6]

        # ---------------------------------------------------------
        # 2. Quaternione (Orientamento)
        # ---------------------------------------------------------
        # Trasformazione analitica inversa:
        # Ruota il frame e corregge l'offset di Yaw di -90 gradi
        qw, qx, qy, qz = x_ned[6:10]
        x_enu[6] = qw
        x_enu[7] = qy
        x_enu[8] = qx
        x_enu[9] = -qz
        
        #c = 0.70710678118  # 1 / sqrt(2)
        #x_enu[6] = c * (qw - qz)      # w
        #x_enu[7] = -c * (qx + qy)     # x
        #x_enu[8] = c * (qx - qy)      # y
        #x_enu[9] = c * (qw + qz)      # z
        
        # Normalizzazione segno w (convenzione per evitare salti)
        if x_enu[6] < 0:
            x_enu[6:10] = -x_enu[6:10]

        # ---------------------------------------------------------
        # 3. Body Rates (Velocità Angolari)
        # ---------------------------------------------------------
        # FRD (Forward-Right-Down) -> FLU (Forward-Left-Up)
        # Roll (X): Forward è uguale in entrambi -> Invariato
        # Pitch (Y): Destra vs Sinistra -> Invertito
        # Yaw (Z): Giù vs Su -> Invertito
        
        p_frd, q_frd, r_frd = x_ned[10:13]
        
        x_enu[10] = p_frd   # p
        x_enu[11] = -q_frd  # q
        x_enu[12] = -r_frd  # r

        return x_enu