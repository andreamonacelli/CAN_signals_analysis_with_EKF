"""
The main execution script that will deal with all the analyses that are required to be performed through the project
"""
import json
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


if __name__ == '__main__':
    input_files_paths = input_filename_generator(BASE_PATH, vehicles_folders, exp_folders)
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

        # Setting the correct lists to be used while filtering the dataframe and defining the motion models
        # Right now it is implemented for ReCAN only, this will be refactored accordingly when OpenDBC and other sources
        # will be integrated in the project
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
            logger.debug(f'Variable {can_var_name} -> Total frames: {len(filtered_df)}')
            logger.debug(f'Variable {can_var_name} -> Calibration frames (80%): {len(calibration_records)}')

            # --- INITIAL EKF SETUP (ONCE PER EACH FILE/EXPERIMENT) ---

            # In order to be more accurate we need to read the first value of the filtered dataframe to set the initial
            # velocity value, otherwise the residual error would spike erroneously
            if len(calibration_records) == 0:
                logger.debug(f'No calibration records found, skipping iteration')
                continue
            first_signal_value = float(calibration_records[0]['value'])
            if current_speed_um == 'kph':
                first_signal_value = (first_signal_value * 0.05) / 3.6

            # Creating an instance of an EKF based on the initial data defined in the respective motion model
            ekf = ExtendedKalmanFilter(dim_x=motion_model.dim_x, dim_z=motion_model.dim_z)
            x_0, P_0 = motion_model.get_initial_state(initial_velocity=first_signal_value)
            ekf.x = x_0
            ekf.P = P_0

            # --- LOOPING OVER THE FRAMES ---

            # For monitoring purposes define a list with the main values obtained from each iteration
            session_traces_list = []
            previous_signal = None
            cumulative_error = 0.0  # Track cumulative error to prevent potential attackers to inject slightly wrong data frames that would get accepted with threshold only check
            iteration_id = 0  # identifier used for tracking reasons
            outfile_path = f'{outfile_dir_path}/test_session_trace_{can_var_name}.json'
            for signal in calibration_records:
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
            # Storing the traces in an external file to ease an evaluation performed later on
            logger.debug(f'Terminated {can_var_name} analysis, now storing results in {outfile_path}...')
            # No need to print a trace if no record has been evaluated
            if iteration_id != 0:
                with open(outfile_path, 'w') as outfile:
                    json.dump(session_traces_list, outfile)
                # To define the boundaries let's find the maximum values (we are going to need this during the "test" stage)
                metrics_df = pd.DataFrame(session_traces_list)
                max_abs_residual = metrics_df['residual_error'].abs().max()
                max_abs_cumulative = metrics_df['cumulative_error'].abs().max()
                print(f'** {can_var_name} BASELINE METRICS **')
                print(f'Var {can_var_name} Maximum Absolute Residual (Single-Frame): {max_abs_residual:.6f} m/s')
                print(f'Var {can_var_name} Maximum Absolute Cumulative Drift:        {max_abs_cumulative:.6f} m/s')
