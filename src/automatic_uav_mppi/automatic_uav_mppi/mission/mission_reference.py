import numpy as np

class SerpentineMission:
    """
    Mission generator for vineyard serpentine coverage.
    Generates a 13D reference state compatible with MPPI.
    
    Improvements:
    - Uses Constant Acceleration for Takeoff, Landing, and Transits.
    - Includes a 'Settle' phase before landing to stabilize the drone.
    """

    def __init__(
        self,
        home=np.array([0.0, 0.0, 0.0]),
        first_row=np.array([-10.0, 20.0]),
        altitude=2.5,
        row_length=20.0,
        row_spacing=2.5,
        num_rows=10,
        v_ref=1.0,
        T_takeoff=4.0,
        T_landing=4.0,
        T_settle=3.0,
    ):
        self.home = np.array(home, dtype=np.float32)
        self.first_row = np.array(first_row, dtype=np.float32)
        self.altitude = altitude
        self.row_length = row_length
        self.row_spacing = row_spacing
        self.num_rows = num_rows
        self.v_ref = v_ref
        self.T_takeoff = T_takeoff
        self.T_landing = T_landing
        self.T_settle = T_settle

        # --- Precomputations ---
        self.R = row_spacing / 2.0
        
        # 1. Serpentine Times
        self.T_row = row_length / v_ref
        self.T_turn = np.pi * self.R / v_ref
        self.T_cycle = self.T_row + self.T_turn
        self.T_serpentine = num_rows * self.T_cycle

        # 2. Transit Times (Constant Acceleration)
        # In a constant acc profile (triangular velocity), V_peak = 2 * V_avg. 
        # To keep V_peak <= v_ref, we need Time >= 2.0 * Dist / v_ref.
        dist_to_vineyard = np.linalg.norm(self.first_row - self.home[:2])
        self.T_transit = 2.0 * dist_to_vineyard / v_ref
        
        # 3. Timeline
        self.t1 = self.T_takeoff
        self.t2 = self.t1 + self.T_transit
        self.t3 = self.t2 + self.T_serpentine
        self.t4 = self.t3 + self.T_transit  # Return time
        self.t5 = self.t4 + self.T_settle   # Hover time
        self.t6 = self.t5 + self.T_landing  # End of Mission

    @staticmethod
    def _constant_acc_step(tau):
        """
        Computes normalized position (0->1) and velocity derivative 
        for a constant acceleration profile.
        """
        tau = np.clip(tau, 0.0, 1.0)
        
        if tau < 0.5:
            # Acceleration Phase (0 -> 0.5)
            # x = 2 * t^2
            # v' = 4 * t
            pos = 2.0 * tau**2
            vel = 4.0 * tau
        else:
            # Deceleration Phase (0.5 -> 1.0)
            # Symmetric relative to the end
            t_rem = 1.0 - tau
            pos = 1.0 - 2.0 * t_rem**2
            vel = 4.0 * t_rem
            
        return pos, vel

    @staticmethod
    def _state(x, y, z, vx, vy, vz):
        """Helper to construct the 13D state vector."""
        return np.array(
            [x, y, z, vx, vy, vz, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    def get_reference(self, t: float) -> np.ndarray:
        """
        Returns a 13D reference state based on mission time t.
        """

        # ---------------- 1. TAKEOFF (Constant Acc.) ----------------
        if t < self.t1:
            t_ramp = 0.4  # secondi
            if t < t_ramp:
                vz_ref = 1.0 * (t / t_ramp)
            else:
                vz_ref = 1.0

            z = self.home[2] + vz_ref * t
            z = min(z, self.altitude)

            return self._state(
                self.home[0],
                self.home[1],
                z,
                0.0,
                0.0,
                vz_ref
            )

        # ---------------- 2. TRANSIT TO VINEYARD (Constant Acc.) ----------------
        elif t < self.t2:
            tau = (t - self.t1) / self.T_transit
            norm_pos, norm_vel = self._constant_acc_step(tau)
            
            dir_vec = self.first_row - self.home[:2]
            
            pos = self.home[:2] + norm_pos * dir_vec
            vel = (dir_vec * norm_vel) / self.T_transit
            
            return self._state(pos[0], pos[1], self.altitude, vel[0], vel[1], 0.0)

        # ---------------- 3. SERPENTINE (Constant Speed) ----------------
        elif t < self.t3:
            ts = t - self.t2
            row_idx = min(int(ts // self.T_cycle), self.num_rows - 1)
            tau = ts - row_idx * self.T_cycle

            y_row = self.first_row[1] + row_idx * self.row_spacing
            direction = 1 if row_idx % 2 == 0 else -1
            
            # --- Straight Segment ---
            if tau < self.T_row:
                s = self.v_ref * tau
                if direction == 1:
                    x = self.first_row[0] + s
                    vx = self.v_ref
                else:
                    x = self.first_row[0] + self.row_length - s
                    vx = -self.v_ref
                return self._state(x, y_row, self.altitude, vx, 0.0, 0.0)

            # --- Turn Segment ---
            # If last row, just finish the straight line
            if row_idx == self.num_rows - 1:
                x_end = self.first_row[0] + (self.row_length if direction == 1 else 0.0)
                vx = direction * self.v_ref
                return self._state(x_end, y_row, self.altitude, vx, 0.0, 0.0)

            t_turn = tau - self.T_row
            theta = np.pi * t_turn / self.T_turn
            x_c = self.first_row[0] + (self.row_length if direction == 1 else 0.0)
            y_c = y_row + self.R

            x = x_c + direction * self.R * np.sin(theta)
            y = y_c - self.R * np.cos(theta)
            vx = direction * self.v_ref * np.cos(theta)
            vy = self.v_ref * np.sin(theta)

            return self._state(x, y, self.altitude, vx, vy, 0.0)

        # ---------------- 4. RETURN TO HOME (Constant Acc.) ----------------
        elif t < self.t4:
            tau = (t - self.t3) / self.T_transit
            norm_pos, norm_vel = self._constant_acc_step(tau)

            # Determine start point (end of serpentine)
            last_row = self.num_rows - 1
            y_last = self.first_row[1] + last_row * self.row_spacing
            direction = 1 if last_row % 2 == 0 else -1
            x_last = self.first_row[0] + (self.row_length if direction == 1 else 0.0)
            
            start = np.array([x_last, y_last])
            end = self.home[:2]
            dir_vec = end - start

            pos = start + norm_pos * dir_vec
            vel = (dir_vec * norm_vel) / self.T_transit

            return self._state(pos[0], pos[1], self.altitude, vel[0], vel[1], 0.0)

        # ---------------- 5. SETTLE (Hover) ----------------
        elif t < self.t5:
            # Velocity must be zero. Position fixed at Home + Altitude.
            return self._state(self.home[0], self.home[1], self.altitude, 0.0, 0.0, 0.0)

        # ---------------- 6. LANDING (Constant Acc.) ----------------
        elif t < self.t6:
            tau = (t - self.t5) / self.T_landing
            norm_pos, norm_vel = self._constant_acc_step(tau)
            
            # Position goes from Altitude -> 0 (1 -> 0 inverted logic)
            z = self.altitude * (1.0 - norm_pos)
            # Velocity is negative
            vz = - (self.altitude * norm_vel) / self.T_landing

            return self._state(self.home[0], self.home[1], z, 0.0, 0.0, vz)

        # ---------------- END OF MISSION ----------------
        else:
            return self._state(self.home[0], self.home[1], 0.0, 0.0, 0.0, 0.0)