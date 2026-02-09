import numpy as np

class Ned2EnuConverter:
    """
    Converts state from NED (North-East-Down) / FRD (Forward-Right-Down)
    to ENU (East-North-Up) / FLU (Forward-Left-Up).
    """
    
    # Permutation Matrix: NED Position -> ENU Position
    # ENU_x (East)  = NED_y
    # ENU_y (North) = NED_x
    # ENU_z (Up)    = -NED_z
    R_matrix = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, -1]])

    @staticmethod
    def ned_to_enu(x_ned):
        """
        Input x_ned: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        Output x_enu: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        """
        x_enu = np.zeros_like(x_ned)

        # 1. Position & Linear Velocity (Vector Rotation)
        x_enu[0:3] = Ned2EnuConverter.R_matrix @ x_ned[0:3]
        x_enu[3:6] = Ned2EnuConverter.R_matrix @ x_ned[3:6]

        # 2. Orientation (Quaternion)
        # We need to rotate the frame: 
        # NED (North=0) -> ENU (East=0) requires a complex rotation, not just a swap.
        qw, qx, qy, qz = x_ned[6:10]
        
        # Standard conversion constant (1/sqrt(2))
        c = 0.70710678118 
        
        # Calculation handles the 90-degree Yaw offset and Z-flip
        x_enu[6] = c * (qw - qz)      # w
        x_enu[7] = -c * (qx + qy)     # x
        x_enu[8] = c * (qx - qy)      # y
        x_enu[9] = c * (qw + qz)      # z

        # Normalize w sign for consistency
        if x_enu[6] < 0:
            x_enu[6:10] = -x_enu[6:10]

        # 3. Body Rates (FRD -> FLU)
        # Roll (X): Forward is same
        # Pitch (Y): Right -> Left (Invert)
        # Yaw (Z): Down -> Up (Invert)
        p_frd, q_frd, r_frd = x_ned[10:13]
        
        x_enu[10] = p_frd   # p
        x_enu[11] = -q_frd  # q
        x_enu[12] = -r_frd  # r

        return x_enu