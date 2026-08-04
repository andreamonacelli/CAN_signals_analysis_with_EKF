import pandas as pd
import numpy as np
import tarfile
import scipy.stats as stats
from scipy.signal import find_peaks


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


def observation_model_h(x):
    """
    Given the current State Model x(t), returns the respective computed observation model h(x)
    :param x: the current state model
    :return: the result of the function that computes the observation model h(x)
    """
    # We just need to fetch the velocity parameter (which is the first in the matrix) from the input state
    return np.array([[x[0, 0]]])


def h_jacobian_speed(x):
    """
    Returns the Jacobian matrix of the observation model h(x) which is denoted in literature as matrix H
    :param x: the current State Model
    :return: the Jacobian matrix of h(x)
    """
    # The partial derivatives of (v) with respect to [v, a] is simply [1, 0]
    return np.array([[1.0, 0.0]])


def check_statistical_distribution(tracing_dataframe):
    """
    Given a Pandas DataFrame holding the data tracings, check which statistical distribution suits best the computed results
    :param tracing_dataframe: the DataFrame that contains the data to be analyzed
    :return: the parameters of the best distribution and a string holding the "code" of the
    statistical distribution that suits best the input data
    """
    # Guard-check to make sure we are going to analyze is not empty
    if len(tracing_dataframe) == 0:
        print('Empty trace file. Cannot perform statistical analysis.')
        return None
    # Extract the steady-state residual errors
    residuals = tracing_dataframe['residual_error'].dropna().values
    # print(f"[DEBUG] Sample Size (N): {len(residuals):,}")

    # First of all, we are going to check if the current distribution is Bimodal/Multimodal
    # To do so, we are going to use the Gaussian KDE of Scipy
    kde = stats.gaussian_kde(residuals)
    # We then evaluate the KDE over the range of our data to get the Y-values of the curve
    # The goal of this is to compare the Y-values in order to find the peaks
    x_grid = np.linspace(residuals.min(), residuals.max(), 200)
    kde_values = kde(x_grid)
    # Find peaks. 'prominence' ensures we only catch real peaks, not tiny noise bumps
    # We set prominence to 10% of the maximum peak height
    prominence_value = float(np.max(kde_values) * 0.1)
    peaks, _ = find_peaks(kde_values, prominence=prominence_value)
    # Finally, we check if the distribution is Bimodal/Multimodal
    if len(peaks) > 1:
        print(f'\n{len(peaks)} PEAKS DETECTED: Distribution is Bimodal/Multimodal. Falling back to KDE.')
        # Return the KDE object as the "params", and 'kde' as the dist name
        return kde, 'kde'

    # If we reach this point, the data is Unimodal (single peak), so we can perform Shapiro-Wilk Test for Normality
    # Note: Scipy warns if N > 5000 because p-values become overly sensitive.
    # We sample 5000 random points within our data-points to get a mathematically fair p-value.
    shapiro_sample_size = min(len(residuals), 5000)
    shapiro_sample = np.random.choice(residuals, shapiro_sample_size, replace=False)
    # shapiro_sample = residuals[:shapiro_sample_size]
    stat_sw, p_value_sw = stats.shapiro(shapiro_sample)
    print(f'Shapiro-Wilk Test -> Statistic: {stat_sw:.4f}, p-value: {p_value_sw:.4e}')

    # Assuming automatically that the best distribution for our data is the normal distribution unless proven differently
    best_dist_name = 'norm'
    best_params = ()
    # Performing the actual check
    if p_value_sw > 0.05:
        # The distribution seems to be actually normal
        best_params = stats.norm.fit(residuals)
    else:
        # The distribution doesn't seem to be a Normal one, thus we need to test it through Kolmogorov-Smirnov distance
        # For the moment we are going to test over some of the most common distributions
        distributions_to_test = ['norm', 'laplace', 'chi2', 'pareto', 'truncnorm']
        # Pre-setting the best_stat (the shortest distance) to infinity to make sure it will get overwritten
        best_ks_stat = float('inf')
        for dist_name in distributions_to_test:
            dist_obj = getattr(stats, dist_name)
            # Scipy automatically calculates the ideal Mean, StdDev and other statistical parameters for our specific data
            params = dist_obj.fit(residuals)
            # KS Test compares the raw data to the theoretical fitted distribution
            stat_ks, p_value_ks = stats.kstest(residuals, dist_name, args=params)
            print(f' - {dist_name.capitalize():<8} KS-Statistic: {stat_ks:.4f} (p-value: {p_value_ks:.4e})')
            # The lowest KS-statistic (the shortest distance) implies the closest mathematical fit
            if stat_ks < best_ks_stat:
                best_ks_stat = stat_ks
                best_dist_name = dist_name
                best_params = params
        print(f'\nThe most suitable statistical distribution is: {best_dist_name.capitalize()}')
    return best_params, best_dist_name
