"""
This module will hold any data-parsing function specific to OpenLKA datasets (common data parsing functions are in the
generic /src/utils/datamanagement.py module)
"""
import pandas as pd
from src.parsers.base_parser import BaseCANParser
from src.models.motion_models import *
from src.utils.mathutils import *


# In this CSV source the vehicle speed is under the vEgo column
# The CSV does not have CAN message IDs since it's a pre-processed CAN dataset


DEFAULT_CAN_ID = '0x123'  # Statically defined custom CAN ID
DEFAULT_VAR_NAME = 'VAR'  # Statically defined custom VAR name


class OpenLKAParser(BaseCANParser):
    """
    This class will hold the necessary functions to parse the data from the ReCAN database
    """
    speed_measure_unit = 'm/s'

    def parse(self, filepath):
        df = pd.DataFrame()
        try:
            # The file passed should be a "ready-to-read" CSV file
            input_df = pd.read_csv(filepath, sep=';', decimal=',', low_memory=False)
            # It's been registered that in the dataset some values are negative (due to sensors noise), since this is
            # physically impossible, we can safely "clean" those values and turn them to 0.0
            clean_velocity_column = input_df['vEgo'].astype('float64').clip(lower=0.0)
            converted_time = pd.to_datetime(input_df['unix_time'], errors='coerce')
            df = pd.DataFrame({
                'id': DEFAULT_CAN_ID,
                'time': converted_time,
                'variable': DEFAULT_VAR_NAME,
                'value': clean_velocity_column
            })
        except FileNotFoundError:
            print(f'ERROR: File {filepath} not found. Please check your source directory!')
        return df

    def set_experiment_params(self, filename):
        speed_ids = [DEFAULT_CAN_ID]
        motion_model = SpeedMotionModel(speed_ids)
        HJacobian = h_jacobian_speed
        Hx = observation_model_h
        return speed_ids, motion_model, HJacobian, Hx
