from filterpy.kalman import ExtendedKalmanFilter
from math import cos, sin, atan2, sqrt
import numpy as np

# Create an extended Kalman filter for a 2D system with non-constant acceleration
def fx(x, dt):
    # State transition function
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Calculate the new state vector
    xdot = x[2]
    ydot = x[3]
    xdotdot = -0.1 * x[2]**3
    ydotdot = -0.1 * x[3]**3
    new_x = np.array([x[0] + dt*xdot + 0.5*dt**2*xdotdot,
                      x[1] + dt*ydot + 0.5*dt**2*ydotdot,
                      xdot + dt*xdotdot,
                      ydot + dt*ydotdot])
    return new_x, F

def hx(x):
    # Measurement function
    return np.array([sqrt(x[0]**2 + x[1]**2), atan2(x[1], x[0])])

# Initialize the EKF
ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
ekf.x = np.array([1, 2, 0, 0])  # initial state vector
ekf.P = np.eye(4)  # initial state covariance
ekf.Q = np.eye(4) * 0.1  # process noise covariance
ekf.R = np.array([[0.1, 0],
                  [0, 0.1]])  # measurement noise covariance

# Run the EKF on some noisy measurements
dt = 0.1  # time step
measurements = [[1.1, 2.0], [2.0, 3.1], [3.1, 4.1], [4.0, 5.2], [5.1, 6.0]]
filtered_states = []

for z in measurements:
    ekf.predict(dt=dt, fx=fx)
    ekf.update(z, hx=hx, HJacobian_at=ekf.jacobian_h, Hx=hx)
    filtered_states.append(ekf.x)

# The filtered_states list contains the filtered estimates of the state
print(filtered_states)
