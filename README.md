# 🚘 CAN Bus Signals Analysis and Prediction using Extended Kalman Filters

## Overview

### 📖 Project Context

This project aims to analyze a stream of signals flowing through a vehicle's CAN Bus in order to predict the next
signal that will be read from the bus and compare it to the actual reading.
Said comparison will allow for an analysis to determine whether there is any kind of anomaly within the data
that is being examined (i.e.: a reading of 15 kph of Vehicle Speed is followed by a reading of 150 kph that occurred
only 10ms later).

To perform analysis and predictions, the project will use an algorithm coming from the State Estimation theory
called **Extended Kalman Filters (EKF)**.
The algorithm requires the definition of the physical motion model of the vehicle in order to define how it will
perform, given the current signal, an accurate prediction for the next signal according to the designed model.

### 🖥️ Technical Details

The project exploits publicly available datasets that hold readings performed over the CAN Bus of real vehicles.
The datasets (currently) used are the likes of [ReCAN](https://github.com/Cyberdefence-Lab-Murcia/ReCAN) and
[CANdid](https://doi.org/10.25909/29068553).
Each dataset, due to its peculiar characteristics, has its own parser in the respective folder. To extend the range
of data to use in order to use the algorithm there are a couple alternative scenarios:
* Make sure that the CAN log format is among the already implemented ones
* Implement a specific parser for the new dataset, documenting also the format that the new parser will deal with

In order to not re-invent the wheel, the project uses the EKF implementation provided by the module 
[**FilterPy**](https://filterpy.readthedocs.io/en/latest/kalman/ExtendedKalmanFilter.html) whose specific theoretical
fundamentals can be found in the [source repository](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb).

---

## 🛠️ Project Structure

```text
CAN_signals_analysis_with_EKF/
├── data/                   # Sample CAN bus logs (in various formats)
├── notebooks/              # Jupyter notebooks for presentations and exploratory analyses
├── outputs/                # Files containing the output of the analytical process
├── src/                    # Source code main folder
│   ├── parsers/            # Parsers that will adjust the format of CAN messages fetched from public datasets
│   ├── models/             # Classes defining the physical models to be used in the EKF
│   ├── utils/              # Generic utilities file that include widely used common functions
│   ├── output_analysis.py  # Script that charts the computed outputs for more human-readable analysis
│   └── main.py             # Main execution script (performing the EKF prediction cycle)
├── requirements.txt        # Python dependencies
└── README.md               # This presentation file!