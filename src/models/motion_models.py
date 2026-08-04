from abc import ABC, abstractmethod
import numpy as np
from filterpy.common import Q_continuous_white_noise


# Useful constants
ALFA_ROMEO_SPEED_IDs = ['0EE', '1F7']
OPEL_CORSA_SPEED_IDs = ['348']
OPEL_CORSA_RPM_IDs = ['0C9']
CAN_LINE = 'can0'


class VehicleMotionModel(ABC):
    """
    The base class that represents the motion model that will be used to define an adequate motion model
    to be fed to the EKF implementation
    """
    def __init__(self, drag_coefficient=0.001, phi=1.0):
        self.drag_coefficient = drag_coefficient
        self.phi = phi

    @property
    @abstractmethod
    def dim_x(self):
        pass

    @property
    @abstractmethod
    def dim_z(self):
        pass

    @abstractmethod
    def get_initial_state(self):
        """Returns the initial startup state (x0, P0)"""
        pass

    @abstractmethod
    def get_transition_stage_data(self, x_old, delta_time):
        """Calculates x_predicted, State Transition matrix F and noise matrix Q which are all needed to perform the transition stage"""
        pass


class SpeedMotionModel(VehicleMotionModel):
    """
    The class that represents the motion model corresponding to the case where we only have the velocity/speed value
    that we can use in order to perform prediction and innovation using EKF
    """
    dim_x = 2
    dim_z = 1

    def __init__(self, CAN_ids):
        super().__init__()
        self.CAN_ids = CAN_ids

    def get_initial_state(self):
        # Assuming the initial state to be stationary (v = 0, a = 0)
        x_0 = np.array([[0.0],
                        [0.0]])
        # Assuming high uncertainty over the values (populating the Covariance Matrix accordingly)
        P_0 = np.array([[10.0, 0.0],
                        [0.0, 10.0]])
        return x_0, P_0

    def get_transition_stage_data(self, x_old, delta_time):
        # Fetching the "old"/current values of speed and acceleration
        v = x_old[0, 0]
        a = x_old[1, 0]
        # Performing prediction leveraging the physical model defined
        v_predicted = v + (a - (self.drag_coefficient * v**2)) * delta_time
        a_predicted = a
        # This is the "manually" predicted value
        x_new = np.array([[v_predicted],
                          [a_predicted]])
        # Defining the State Transition Matrix
        F = np.array([
            [1.0 - (2.0 * self.drag_coefficient * v_predicted * delta_time), delta_time],
            [0.0, 1.0]
        ])
        # Defining the noise matrices (respectively Q=process noise and R=measurement noise)
        Q = Q_continuous_white_noise(dim=2, dt=delta_time, spectral_density=self.phi)
        R = np.array([[0.5]])
        return x_new, F, Q, R


# In case the SpeedMotionModel works, it might be worth giving it a shot to try and define a physical model to
# include predictions over RPM values in the picture as well (to be agreed)
class SpeedRPMMotionModel(VehicleMotionModel):
    """
    [Work-In-Progress] The class that represents the motion model corresponding to the case where we have both the
    velocity/speed value and the RPM reading that we can use in order to perform prediction and innovation using EKF
    """
    CAN_ids = OPEL_CORSA_SPEED_IDs + OPEL_CORSA_RPM_IDs
    dim_x = 3
    dim_z = 1

    def get_initial_state(self):
        pass

    def get_transition_stage_data(self, x_old, delta_time):
        pass
