import numpy as np

class Enu2NedConverter:
    @staticmethod
    def enu_to_ned(x_enu):
        """
        Converte uno stato da ENU a NED
        x_enu: array [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
        Ritorna: array NED con:
            [pos_ned, vel_ned, quat_ned, p,q,r]
        """
        R = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, -1]])

        x = np.zeros_like(x_enu)
        # posizione e velocità
        x[0:3] = R @ x_enu[0:3]
        x[3:6] = R @ x_enu[3:6]
        # quaternion
        qw, qx, qy, qz = x_enu[6:10]
        x[6:10] = [qw, qy, qx, -qz]
        # velocità angolare (p, q, r)
        x[10:13] = x_enu[10:13]
        return x
    

    #@staticmethod
    #def enu_to_ned(x_enu):
    #    R = np.array([[0,1,0],[1,0,0],[0,0,-1]])
    #    x = np.zeros_like(x_enu)
    #    x[0:3] = R @ x_enu[0:3]
    #    x[3:6] = R @ x_enu[3:6]
    #    qw,qx,qy,qz = x_enu[6:10]
    #    x[6:10] = [qw, qy, qx, -qz]
    #    x[10:13] = x_enu[10:13]
    #    return x
