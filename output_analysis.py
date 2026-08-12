"""
The script that fetches the output data and displays charts and metrics useful for results analyses
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.generic import *
from src.utils.datamanagement import *
from src.utils.mathutils import *


# PARAMETERS SETUP
MAX_EXPERIMENT_COUNT = 10
BASE_PATH = 'outputs'
vehicles_folders = os.listdir(BASE_PATH)
exp_folders = [f'Exp-{i}' for i in range(1, MAX_EXPERIMENT_COUNT)]
can_var_names = ['BY_0', 'HA_0', 'HA_1', 'HA_2', 'NI_0', 'NI_1', 'VAR']  # It would be better to make this dynamic


if __name__ == '__main__':
    traces_files = traces_filenames_generator(BASE_PATH, vehicles_folders, exp_folders, can_var_names)
    for f in traces_files:
        print(f'\n=========== NOW CHARTING TRACE FILE {f} ===========')
        tracing_dict = json.load(open(f, 'r'))
        tracing_df = pd.DataFrame(tracing_dict)

        # Eventually loading the metrics dictionary, creating it otherwise
        vehicle_name, experiment_id, var_name = get_info_from_path(f)
        metrics_filepath = f'{BASE_PATH}/{vehicle_name}/{experiment_id}/metrics.json'
        if os.path.exists(metrics_filepath):
            metrics_dict = json.load(open(metrics_filepath, 'r'))
            metrics_dict[var_name] = {}
        else:
            metrics_dict = {}
        # By loading/creating the metrics dictionary the way we just did we are able to overwrite the metrics file without worrying

        if len(tracing_df) > 0:
            steady_state_errors = tracing_df['residual_error']

            # Check which statistical distribution suits best our data
            best_params, best_dist_name = check_statistical_distribution(tracing_df)

            fig, axes = plt.subplots()

            sns.histplot(steady_state_errors, bins=100, kde=False, ax=axes, color='blue', stat='density')
            xmin, xmax = axes.get_xlim()
            x = np.linspace(xmin, xmax, 100)

            # If the distribution is Bimodal/Multimodal we should handle the respective object accordingly
            if best_dist_name == 'kde':
                kde_obj = best_params
                p = kde_obj(x)
                # Since KDE doesn't have a theoretical mean/std, calculate from the raw data
                mu = steady_state_errors.mean()
                std = steady_state_errors.std()
                axes.plot(x, p, 'k', linewidth=2, label=f'Non-Parametric KDE')
                axes.set_title('Innovation Residual Distribution (Multimodal)')
            else:
                # Leveraging the custom statistical check that has been defined in the utilities.py file
                best_dist_obj = getattr(stats, best_dist_name)
                p = best_dist_obj.pdf(x, *best_params)
                mu = best_dist_obj.mean(*best_params)
                std = best_dist_obj.std(*best_params)
                dist_label = '$\\mathcal{N}$' if best_dist_name == 'norm' else best_dist_name.capitalize()
                axes.plot(x, p, 'k', linewidth=2, label=f'Ideal {dist_label} ($\\mu$={mu:.4f}, $\\sigma$={std:.4f})')
                axes.set_title('Innovation Residual Distribution (Unimodal)')

            # Common parameters and plotting the chart
            axes.set_xlabel('Kinematic Discrepancy (m/s)')
            axes.set_ylabel('Density')
            axes.legend()
            plt.tight_layout()
            plt.show()

            if best_dist_name == 'kde':
                metrics_title = 'Multimodal'
            else:
                metrics_title = best_dist_name.capitalize()
            print(f'** {metrics_title} DISTRIBUTION METRICS **')
            print(f'Mean Error (mu): {mu:.6f} m/s')
            print(f'Standard Dev (sigma): {std:.6f} m/s')

            # Updating the metrics dictionary and storing it into the respective file
            metrics_dict[var_name] = {
                'distribution': metrics_title,
                'mean': mu,
                'std_deviation': std,
                'max_cumulative_error': tracing_df['cumulative_error'].max()
            }
            with open(metrics_filepath, 'w') as metrics_file:
                json.dump(metrics_dict, metrics_file)
