"""
The main execution script that will deal with all the analyses that are required to be performed through the project
"""
import datetime
import json

import pandas as pd

from src.parsers.factories import ParserFactory

from filterpy.kalman import ExtendedKalmanFilter
from src.utils.generic import *
from src.utils.datamanagement import *
from src.parsers.recan import *

# Logger setup
logger = logging.getLogger(__name__)


# SETUP STAGE: Fetching the input filenames
MAX_EXPERIMENT_COUNT = 10
BASE_PATH = 'data'
vehicles_folders = os.listdir(BASE_PATH)
exp_folders = [f'Exp-{i}' for i in range(1, MAX_EXPERIMENT_COUNT)]


def run_ekf_loop(records, motion_model, HJacobian, Hx, current_speed_um):
    """
    Functions that handles the actual EKF loop for a given set of parameters that are needed to adequately perform the
    needed prediction and update steps.
    :param records: a list holding the records over which the loop is supposed to iterate over.
    :param motion_model: the model that hold Transition functions and Noise matrices to be fed to the EKF.
    :param HJacobian: the Jacobian matrix of the observation model h(x).
    :param Hx: the respective observation model h(x).
    :param current_speed_um: the unit measure of the values within the record, useful for conversion purposes
    :return: the session traces for the loop
    """
    # If the records list is empty then return an empty trace
    if not records:
        return []
    # Set the initial speed value accordingly (this value may differ from one record set to another)
    first_signal_value = float(records[0]['value'])
    if current_speed_um == 'kph':
        first_signal_value = (first_signal_value * 0.05) / 3.6
    # Setting up the EKF according to the data being used
    ekf = ExtendedKalmanFilter(dim_x=motion_model.dim_x, dim_z=motion_model.dim_z)
    x_0, P_0 = motion_model.get_initial_state(initial_velocity=first_signal_value)
    ekf.x = x_0
    ekf.P = P_0
    # Preparing monitoring variables
    session_traces_list = []
    previous_signal = None
    cumulative_error = 0.0  # Track cumulative error to prevent potential attackers to inject slightly wrong data frames that would get accepted with threshold only check
    iteration_id = 0  # identifier used for tracking reasons
    # PERFORMING THE ACTUAL LOOP
    for signal in records:
        session_trace = {'iteration_id': iteration_id}

        # --- READING THE ACTUAL NEXT VALUE (and converting it in m/s if necessary) ---
        signal_value = float(signal['value'])
        if current_speed_um == 'kph':
            # Perform the conversion based on the measurement unit noted in the parser
            signal_value = (signal_value * 0.05) / 3.6
        z_k = np.array([[signal_value]])
        session_trace['timestamp'] = str(signal['time'])
        session_trace['signal_value'] = signal_value

        # If it's the first signal, we must assume delta_time as 0
        if previous_signal is None:
            delta_time = 0
        else:
            delta_time = (signal['time'] - previous_signal['time']).total_seconds()
        # Guard against out-of-order serial logging artifacts (time will not go backwards, thus delta_time can never be negative)
        if delta_time < 0:
            continue
        session_trace['delta_time'] = delta_time

        # --- USING THE EKF TO PERFORM THE PREDICTION (PREDICT THE CURRENT VALUE BASED ON THE PREVIOUS STATE) ---
        # We can now extract the other matrices from the motion model to update the EKF
        x_new, F, Q, R = motion_model.get_transition_stage_data(ekf.x, delta_time)
        ekf.x = x_new
        ekf.F = F
        ekf.Q = Q
        ekf.R = R
        # Updating the predicted Covariance Matrix P according to the defined physical model
        ekf.P = np.dot(ekf.F, ekf.P).dot(ekf.F.T) + ekf.Q

        # Performing the correction/update step for the EKF
        # This step actually updates the data held in the EKF by performing the Innovation Stage of the filter
        ekf.update(z=z_k, HJacobian=HJacobian, Hx=Hx)
        session_trace['predicted_next_value'] = ekf.x[0, 0]

        # --- PREDICTION EVALUATION AND ITERATION LOGGING ---
        # Note: we can use the residual, however we should refine our approach for a bit because we would need to consider
        # the EKF prediction as the Ground Truth, and we should then compare the received data to see whether it falls within the right distribution
        y_k = float(ekf.y[0, 0])  # Innovation Residual - Represents the prediction error
        cumulative_error += y_k  # Updating the cumulative error with the residual for the current signal
        variance_s = float(ekf.S[0, 0])  # Innovation Covariance Matrix S, which we will use to define a threshold
        evaluation_threshold = 3.0 * np.sqrt(variance_s)  # The threshold to identify errors over a single iteration
        session_trace['residual_error'] = ekf.y[0, 0]
        session_trace['evaluation_threshold'] = evaluation_threshold
        session_trace['cumulative_error'] = cumulative_error

        # --- PREPARING FOR NEXT ITERATION ---
        # Once all the evaluations are correctly done, save the current signal into the "previous_signal" variable in order to compute delta_time properly
        previous_signal = signal
        iteration_id += 1
        session_traces_list.append(session_trace)
    return session_traces_list


if __name__ == '__main__':
    input_files_paths = input_filename_generator(BASE_PATH, vehicles_folders, exp_folders)

    # Initializing the full report file (useful to analyze the results of the algorithm)
    # Every time we run the algorithm we clear the previous report file, to keep it we should do a manual backup
    report_file_path = f'outputs/full_report.txt'
    with open(report_file_path, 'w') as report_file:
        report_file.write(f'##### ALGORITHM FULL REPORT - Launched on date: {datetime.datetime.now()} #####')

    # For each file we should loop over the whole dataset
    for exp_file in input_files_paths:
        stats_dict = {'file': exp_file}
        print(f'!!!---- NOW PROCESSING {exp_file} ----!!!')
        # The first thing to do is converting the parsed file into a pandas dataframe leveraging the method in utilities
        # We will get the dataframe without binary readings since they are not very interesting in our context

        # Based on the experiment file currently under examination, fetch the data into a Pandas DataFrame
        source_dataset = exp_file.split('/')[1].split('-')[0]
        parser = ParserFactory.get_parser(source_dataset)
        if parser is None:
            print(f'No parser exists for file {exp_file}')
            continue
        df = parser.parse(exp_file)

        stats_dict['total_file_records'] = len(df)
        logger.debug(f'The file {exp_file} has been converted to dataframe and has {len(df)} rows (non-binary records)')

        # Defining the output directory path (and creating it in case it doesn't exist)
        outfile_dir_path = os.path.split(exp_file)[0].replace('data', 'outputs')
        if not os.path.exists(outfile_dir_path):
            os.makedirs(outfile_dir_path)

        # Setting the correct lists to be used while filtering the dataframe and defining the motion models.
        # Right now it is implemented for ReCAN only, this will be refactored accordingly when OpenDBC and other sources
        # will be integrated in the project.
        # speed_ids, motion_model, HJacobian, Hx = set_experiment_params(exp_file)
        speed_ids, motion_model, HJacobian, Hx = parser.set_experiment_params(exp_file)
        current_speed_um = parser.speed_measure_unit

        can_variables = get_var_names(df)
        # If no "variables" are found within the dataframe (thus we are likely to be working with a non-ReCAN source)
        # we just place a "dummy" variable in the list in order to enter the loop
        if len(can_variables) == 0:
            can_variables.append('VAR')
        logger.debug(f'-> CAN variables in current file: {can_variables}')

        # We must perform the prediction separately for every single variable
        for can_var_name in can_variables:
            print(f'======= CURRENTLY ANALYZING VARIABLE {can_var_name} for file {exp_file} =======')

            # --- DATAFRAME PREPARATION ---

            # We are going to use the 80/20 method (80% of data used for Calibration of the filter, 20% of the data used for later testing)
            # As of now we are just keeping the first 80% of the values to enhance Calibration even more we may have to take the median 80% values
            # In case we're working with a ReCAN source we must deal with the variables, otherwise we should just deal
            # with the correct IDs
            if can_var_name == 'VAR':
                if len(speed_ids) == 0:
                    filtered_df = df  # in case the IDs are not specified we can assume that the whole dataset is filled with speed-related messages
                else:
                    filtered_df = df[(df['id'].isin(speed_ids))]
            else:
                filtered_df = df[(df['id'].isin(speed_ids)) & (df['variable'] == can_var_name)]
            filtered_df.sort_values(by=['time'], inplace=True)
            data_records = filtered_df.to_dict('records')
            split_index = int(len(data_records) * 0.8)
            calibration_records = data_records[:split_index]
            validation_records = data_records[split_index:]
            logger.debug(f'Variable {can_var_name} -> Total frames: {len(filtered_df)}')
            logger.debug(f'Variable {can_var_name} -> Calibration frames (80%): {len(calibration_records)}')
            logger.debug(f'Variable {can_var_name} -> Validation frames (20%): {len(validation_records)}')

            # --- INITIAL SETUP (ONCE PER EACH FILE/EXPERIMENT) ---

            # In order to be more accurate we need to read the first value of the filtered dataframe to set the initial
            # velocity value, otherwise the residual error would spike erroneously
            if len(calibration_records) == 0:
                logger.debug(f'No calibration records found, skipping iteration')
                continue
            outfile_path = f'{outfile_dir_path}/test_session_trace_{can_var_name}.json'

            # --- PERFORMING 80/20 CALIBRATION AND VALIDATION TECHNIQUE ---

            # --- CALIBRATION STAGE (80%) ---
            session_traces_list = run_ekf_loop(
                calibration_records,
                motion_model,
                HJacobian,
                Hx,
                current_speed_um
            )
            # Storing the traces in an external file to ease an evaluation performed later on
            logger.debug(f'Terminated {can_var_name} analysis, now storing results in {outfile_path}...')
            # No need to print a trace if no record has been evaluated
            if len(session_traces_list) != 0:
                with open(outfile_path, 'w') as outfile:
                    json.dump(session_traces_list, outfile)
                # To define the boundaries let's find the maximum values (we are going to need this during the validation stage)
                metrics_df = pd.DataFrame(session_traces_list)
                max_abs_residual = metrics_df['residual_error'].abs().max()
                max_abs_cumulative = metrics_df['cumulative_error'].abs().max()
                print(f'** {can_var_name} BASELINE METRICS **')
                print(f'Var {can_var_name} Maximum Absolute Residual (Single-Frame): {max_abs_residual:.6f} m/s')
                print(f'Var {can_var_name} Maximum Absolute Cumulative Drift:        {max_abs_cumulative:.6f} m/s')
            else:
                # If we weren't able to perform the calibration stage we will surely not be able to perform validation
                logger.debug(f'Skipping validation loop due to absence of calibration data...')
                continue

            # In the validation stage we might want to analyze signals and identify potential anomalies.
            # An effective way to do it would be to define the statistical distribution of the data and then take the
            # 99.8th percentile (regardless of the distribution) to define the threshold of the values
            mu_calib, sigma_calib, anomaly_threshold, dist_name = compute_anomaly_threshold(session_traces_list, 0.998)

            # --- VALIDATION/TEST STAGE (20%) ---
            validation_traces_list = run_ekf_loop(
                validation_records,
                motion_model,
                HJacobian,
                Hx,
                current_speed_um
            )
            logger.debug(f'Terminated the loop over the validation records, now performing data analysis')
            if len(validation_traces_list) != 0:
                validation_df = pd.DataFrame(validation_traces_list)
                # To flag a signal as anomaly we are, for now, performing a simple check that will be useful to output
                # the data and perform further detailed analysis over it
                validation_df['is_anomaly'] = (validation_df['residual_error'].abs() > anomaly_threshold)
                anomalies_count = validation_df['is_anomaly'].sum()
                print(f'** {can_var_name} VALIDATION/TEST METRICS **')
                print(f'Var {can_var_name} Total Test Frames Analyzed: {len(validation_df)}')
                print(f'Var {can_var_name} Readings flagged as Anomalies: {anomalies_count}')
                print(f'Var {can_var_name} Anomaly Rate: {(anomalies_count / len(validation_df)) * 100:.2f}%')
                # We then save these results in an output file, the same way we did with the calibration data
                validation_outfile_path = f'{outfile_dir_path}/validation_session_trace_{can_var_name}.json'
                with open(validation_outfile_path, 'w') as outfile:
                    json.dump(validation_traces_list, outfile)

            # Now we are going to write the full experiment report in the file
            report_text = f"""
            --- EXPERIMENT FILE: {exp_file} - VARIABLE: {can_var_name} ---
            [Calibration Step, performed over 80% of total records] -> No. of records: {len(calibration_records)}
            - Statistical Distribution of Residuals:    {dist_name.capitalize()}
            - Mean value of Residuals (mu):             {mu_calib}
            - Standard Deviation of Residuals (sigma):  {sigma_calib}
            - Anomaly threshold:                        {anomaly_threshold}
            - Max. Cumulative Drift:                    {max_abs_cumulative}
            [Validation Step, performed over 20% of total records] -> No. of records: {len(validation_records)}
            - Signals flagged as anomalies:             {anomalies_count}
            - Anomaly rate (anomalies/total frames %):  {(anomalies_count / len(validation_df)) * 100}
            """
            with open(report_file_path, 'a') as report_outfile:
                report_outfile.write(report_text)
