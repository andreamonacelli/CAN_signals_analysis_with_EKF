import pandas as pd
import tarfile


def convert_to_dataframe(tar_file_path):
    """
    Given a CSV file packed into a .tar.gz archive, returns a Pandas DataFrame properly parsed from it
    :param tar_file_path: the absolute path to a .tar.gz file
    :return: a Pandas DataFrame containing the parsed data
    """
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


# TO BE PARAMETRIZED IN ORDER TO MAKE THE VALUES "UNIVERSAL"
def get_var_names(raw_dataframe):
    """
    Given a DataFrame holding CAN readings, returns a list of all the available variable names that ARE NOT associated
    to binary, crc or counter datatype (thus considering only word, halfword, byte or nibble readings)
    :param raw_dataframe: the dataframe to be analyzed
    :return: a list holding the variable names
    """
    filtered_df = raw_dataframe[raw_dataframe['datatype'].isin(['word', 'halfword', 'byte', 'nibble'])]
    can_var_names = filtered_df['variable'].unique().tolist()
    return can_var_names
