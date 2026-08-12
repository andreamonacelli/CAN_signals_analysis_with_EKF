import pandas as pd
import tarfile


def get_info_from_path(filepath):
    """
    Given a filepath (as string), this function parses the Vehicle code, the Experiment ID and the Variable Name for the current file
    :param filepath: the path of the file currently under analysis
    :return: the Vehicle Code, the Experiment ID and the Variable name as strings
    """
    sections = filepath.split('/')
    if sections[0] == '.':
        vehicle_code = sections[2]
        experiment_id = sections[3]
        file_name = sections[4]
    else:
        vehicle_code = sections[1]
        experiment_id = sections[2]
        file_name = sections[3]
    var_name = file_name.split('.')[0][-4:]
    return vehicle_code, experiment_id, var_name


def get_var_names(raw_dataframe):
    """
    Given a DataFrame holding CAN readings, returns a list of all the available variable names that ARE NOT associated
    to binary, crc or counter datatype (thus considering only word, halfword, byte or nibble readings).
    Note that this will actually return a meaningful list only for ReCAN-based dataframes.
    :param raw_dataframe: the dataframe to be analyzed
    :return: a list holding the variable names (an empty list if the input dataframe doesn't have the respective column)
    """
    can_var_names = []
    if 'datatype' in raw_dataframe.columns:
        filtered_df = raw_dataframe[raw_dataframe['datatype'].isin(['word', 'halfword', 'byte', 'nibble'])]
        can_var_names = filtered_df['variable'].unique().tolist()
    return can_var_names
