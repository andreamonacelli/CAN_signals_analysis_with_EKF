"""
This module will hold any data-parsing function specific to ReCAN databases (common data parsing functions are in the
generic /src/utils/datamanagement.py module)
"""
from src.models.motion_models import *
from src.utils.mathutils import *


ALFA_ROMEO_SPEED_IDs = ['0EE', '1F7']
OPEL_CORSA_SPEED_IDs = ['348']
OPEL_CORSA_RPM_IDs = ['0C9']  # To keep only if it will be decided to proceed with RPMs as well
CAN_LINE = 'can0'


def set_experiment_params(filename):
    """
    Given the filename, returns the parameters that need to be used in the experiment according to the model
    :param filename: a string representing the filename for the current experiment
    :return: the list of the correct CAN message IDs, Motion Model implementation and correct analytical functions to
    use within the EKF implementation
    """
    if 'C-1' in filename:
        speed_ids = ALFA_ROMEO_SPEED_IDs
    else:
        speed_ids = OPEL_CORSA_SPEED_IDs
    motion_model = SpeedMotionModel(speed_ids)
    HJacobian = h_jacobian_speed
    Hx = observation_model_h
    return speed_ids, motion_model, HJacobian, Hx
