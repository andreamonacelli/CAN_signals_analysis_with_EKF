import os
import logging
from pyprojroot import here

# Logger setup
logger = logging.getLogger(__name__)


def input_filename_generator(base_path, vehicles_folders, exp_folders):
    file_paths = []
    for vehicle_name in vehicles_folders:
        for exp_folder in exp_folders:
            root_path = here()
            folder_path = root_path / f'{base_path}/{vehicle_name}/{exp_folder}/'
            if os.path.exists(folder_path):
                files_in_path = os.listdir(folder_path)
                for filename in files_in_path:
                    path = f'{base_path}/{vehicle_name}/{exp_folder}/{filename}'
                    if os.path.exists(path):
                        file_paths.append(path)
    logger.debug(f'Generated input filepaths: {file_paths}')
    return file_paths


def traces_filenames_generator(base_path, vehicles_folders, exp_folders, var_names):
    traces_files = []
    for vehicle_name in vehicles_folders:
        for exp_folder in exp_folders:
            for var_name in var_names:
                path = f'{base_path}/{vehicle_name}/{exp_folder}/test_session_trace_{var_name}.json'
                if os.path.exists(path):
                    traces_files.append(path)
    logger.debug(f'Generated traces filepaths: {traces_files}')
    return traces_files
