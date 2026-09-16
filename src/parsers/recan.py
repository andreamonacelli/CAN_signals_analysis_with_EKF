"""
This module will hold any data-parsing function specific to ReCAN databases (common data parsing functions are in the
generic /src/utils/datamanagement.py module)
"""
import tarfile
import pandas as pd
from src.parsers.base_parser import BaseCANParser
from src.models.motion_models import *
from src.utils.mathutils import *


# The format handled by this parser is the following:
# 2018-07-26 15:15:58.643918,0DE,can0,halfword,HA_0,7843
# time,id,can,datatype,variable,value


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


class ReCANParser(BaseCANParser):
    """
    This class will hold the necessary functions to parse the data from the ReCAN database
    """
    speed_measure_unit = 'kph'

    def parse(self, tar_file_path):
        df = pd.DataFrame()
        try:
            with tarfile.open(tar_file_path, "r:gz") as tar:
                members = tar.getmembers()
                f = tar.extractfile(members[0])
                if f is not None:
                    df = pd.read_csv(f, sep=',', parse_dates=['time'], low_memory=False)
                    # Based on the description of the binary data type given in the paper we can keep them out of the dataframe
                    df = df[df['datatype'] != 'binary']
        except FileNotFoundError:
            print(f'ERROR: File {tar_file_path} not found. Please check your source directory!')
        return df

    def set_experiment_params(self, filename):
        if 'C-1' in filename:
            speed_ids = ALFA_ROMEO_SPEED_IDs
        else:
            speed_ids = OPEL_CORSA_SPEED_IDs
        motion_model = SpeedMotionModel(
            CAN_ids=speed_ids,
            drag_coefficient=0.0001,
            phi=5.0,
            measurement_noise=0.5
        )
        HJacobian = h_jacobian_speed
        Hx = observation_model_h
        return speed_ids, motion_model, HJacobian, Hx
