import numpy as np


class SerpentineMission:
    """
    Mission generator for vineyard serpentine coverage.
    Generates a 13D reference state compatible with MPPI.
    """

    def __init__(
        self,
        home=np.array([0.0, 0.0, 0.0]),
        first_row=np.array([-10.0, 20.0]),
        altitude=2.5,
        row_length=20.0,
        row_spacing=2.5,
        num_rows=10,
        v_ref=2.0,
        T_takeoff=3.0,
        T_landing=3.0,
    ):
        self.home = home
        self.first_row = first_row
        self.altitude = altitude
        self.row_length = row_length
        self.row_spacing = row_spacing
        self.num_rows = num_rows
        self.v_ref = v_ref
        self.T_takeoff = T_takeoff
        self.T_landing = T_landing

        # --- Precomputations ---
        self.R = row_spacing / 2.0
        self.T_row = row_length / v_ref
        self.T_turn = np.pi * self.R / v_ref
        self.T_cycle = self.T_row + self.T_turn
        self.T_serpentine = num_rows * self.T_cycle
        self.T_to_vineyard = np.linalg.norm(first_row - home[:2]) / v_ref

        self.t1 = T_takeoff
        self.t2 = self.t1 + self.T_to_vineyard
        self.t3 = self.t2 + self.T_serpentine
        self.t4 = self.t3 + self.T_to_vineyard
        self.t5 = self.t4 + T_landing

    def get_reference(self, t: float) -> np.ndarray:
        """
        Returns a 13D reference state:
        [x,y,z, vx,vy,vz, qw,qx,qy,qz, p,q,r]
        """

        # ---------------- TAKEOFF ----------------
        if t < self.t1:
            z = self.altitude * (t / self.T_takeoff)
            vz = self.altitude / self.T_takeoff
            return self._state(self.home[0], self.home[1], z, 0, 0, vz)

        # ---------------- TRANSIT ----------------
        elif t < self.t2:
            tau = (t - self.t1) / self.T_to_vineyard
            pos = self.home[:2] + tau * (self.first_row - self.home[:2])
            vel = self.v_ref * (self.first_row - self.home[:2])
            vel /= np.linalg.norm(vel)
            return self._state(pos[0], pos[1], self.altitude, vel[0], vel[1], 0)

        # ---------------- SERPENTINE ----------------
        elif t < self.t3:
            ts = t - self.t2
            row_idx = min(int(ts // self.T_cycle), self.num_rows - 1)
            tau = ts - row_idx * self.T_cycle

            y_row = self.first_row[1] + row_idx * self.row_spacing
            direction = 1 if row_idx % 2 == 0 else -1

            # Straight
            if tau < self.T_row:
                s = self.v_ref * tau
                x = self.first_row[0] + s if direction == 1 else self.first_row[0] + self.row_length - s
                vx = direction * self.v_ref
                return self._state(x, y_row, self.altitude, vx, 0, 0)

            # Turn
            if row_idx == self.num_rows - 1:
                x = self.first_row[0] + (self.row_length if direction == 1 else 0)
                return self._state(x, y_row, self.altitude, direction * self.v_ref, 0, 0)

            t_turn = tau - self.T_row
            theta = np.pi * t_turn / self.T_turn
            x_c = self.first_row[0] + (self.row_length if direction == 1 else 0)
            y_c = y_row + self.R

            x = x_c + direction * self.R * np.sin(theta)
            y = y_c - self.R * np.cos(theta)
            vx = direction * self.v_ref * np.cos(theta)
            vy = self.v_ref * np.sin(theta)

            return self._state(x, y, self.altitude, vx, vy, 0)

        # ---------------- RETURN ----------------
        elif t < self.t4:
            tau = (t - self.t3) / self.T_to_vineyard
            tau = np.clip(tau, 0, 1)

            last_row = self.num_rows - 1
            y_last = self.first_row[1] + last_row * self.row_spacing
            direction = 1 if last_row % 2 == 0 else -1
            x_last = self.first_row[0] + (self.row_length if direction == 1 else 0)

            pos = np.array([x_last, y_last]) + tau * (self.home[:2] - np.array([x_last, y_last]))
            vel = self.v_ref * (self.home[:2] - np.array([x_last, y_last]))
            vel /= np.linalg.norm(vel)

            return self._state(pos[0], pos[1], self.altitude, vel[0], vel[1], 0)

        # ---------------- LANDING ----------------
        elif t < self.t5:
            tau = (t - self.t4) / self.T_landing
            z = self.altitude * (1 - tau)
            return self._state(self.home[0], self.home[1], z, 0, 0, -self.altitude / self.T_landing)

        # ---------------- END ----------------
        return self._state(self.home[0], self.home[1], self.home[2], 0, 0, 0)

    @staticmethod
    def _state(x, y, z, vx, vy, vz):
        return np.array(
            [x, y, z, vx, vy, vz, 1, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
