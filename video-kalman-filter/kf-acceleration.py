from filterpy.kalman import KalmanFilter
import numpy as np

# Create a 2D Kalman filter with a constant acceleration model
kf = KalmanFilter(dim_x=6, dim_z=2)
dt = 1.0  # time step
kf.F = np.array([[1., 0., dt, 0., 0.5*dt**2, 0.],
                 [0., 1., 0., dt, 0., 0.5*dt**2],
                 [0., 0., 1., 0., dt, 0.],
                 [0., 0., 0., 1., 0., dt],
                 [0., 0., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 0., 1.]])  # state transition matrix
kf.H = np.array([[1., 0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0., 0.]])  # measurement matrix
kf.Q = np.eye(6) * 0.1  # process noise covariance
kf.R = np.eye(2) * 1.0  # measurement noise covariance
kf.P = np.eye(6)  # initial state covariance

# Run the filter on some noisy measurements
measurements = [[1.1, 2.0], [2.0, 3.1], [3.1, 4.1], [4.0, 5.2], [5.1, 6.0]]
filtered_states = []

for z in measurements:
    kf.predict()
    kf.update(z)
    filtered_states.append(kf.x)

# The filtered_states list contains the filtered estimates of the state
print(filtered_states)
