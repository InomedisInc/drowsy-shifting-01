import os
import pandas as pd
from importlib_resources import files
from pygments.unistring import combine
from tqdm import tqdm
from colorama import Fore, Style, init
import numpy as np

# Initialize colorama for Windows support
init()

# Constants
ROOT_DIR = "F:\\Data_Driving\\Recordings"
# DESTINATION_DIR = ""  # Specify the destination directory

PERCLOS_FILENAME = "perclos_data"
ECG_FILENAME = "biopac"
MODEL_SAVE_PATH = os.path.join('data', "models", "model.h5")  # Added model save path

# ----- perclos.py -----
from perclos import process_perclos_and_yawning
from perclos import process_blinks


# ----- processing.py -----
from processing import detect_r_peaks_and_calculate_rr
from processing import calculate_respiratory_rate
from processing import create_crash_derivative_dataframe

# ----- crashes.py -----
from crashes import add_crashes_columns
from crashes import process_simulator_data

# ---- models_shifting.py -----
from models_shifting import train_with_time_shifts

# ----- models.py -----
from models import train_crash_detection_model
from models import predict_crashes
from models import evaluate_predictions
from models import save_model_summary
from models import export_predictions
from models import plot_predictions


def get_patient_list(root_dir):
    return sorted([f for f in os.listdir(root_dir) if f.startswith('P0') and not f.endswith('lnk')])

def get_session_list(root_dir, patient):
    return [f for f in os.listdir(os.path.join(root_dir, patient)) if f.startswith(f'{patient}_S0')]


def process_session(root_dir, destination_dir, patient, session):
    aligned_folder = f"{session}_aligned"
    ecg_file = os.path.join(root_dir, patient, session, aligned_folder, f"{session}_{ECG_FILENAME}.csv")

    print(f"{Fore.MAGENTA}Processing {patient}/{session}...{Style.RESET_ALL}")
    print(f"{Fore.LIGHTYELLOW_EX}ECG file: {ecg_file}{Style.RESET_ALL}")

    # Read ECG data
    rr_intervals_path = os.path.join(destination_dir, 'data', patient, session, 'rr_intervals')
    rr_intervals_file = os.path.join(rr_intervals_path, session, "rr_intervals.csv")

    if not os.path.exists(rr_intervals_file):
        ecg_data = pd.read_csv(ecg_file, engine='python')
        ecg_data['time'] = ecg_data['time'].astype(float)

        # Process respiratory data
        respiratory_data = None
        respiratory_rate_df = None

        if 'Biopac_0' in ecg_data.columns:

            respiratory_data = ecg_data[['time', 'Biopac_0']].rename(columns={'Biopac_0': 'respiratory'})

            resp_rate_dir = os.path.join(destination_dir, 'data', patient, session, 'respiratory_rate')
            os.makedirs(resp_rate_dir, exist_ok=True)
            resp_rate_file = os.path.join(resp_rate_dir, f"{session}_respiratory_rate.csv")

            if os.path.exists(resp_rate_file):
                respiratory_rate_df = pd.read_csv(resp_rate_file)
                print(f"{Fore.GREEN}Respiratory rate data found{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Respiratory data extracted successfully{Style.RESET_ALL}")

                respiratory_rate_df = calculate_respiratory_rate(respiratory_data)
                if not respiratory_rate_df.empty:
                    respiratory_rate_df.to_csv(resp_rate_file, index=False)

        # Process ECG data
        ecg_data = ecg_data[['time', 'Biopac_2']].rename(columns={'Biopac_2': 'ecg'})

        rr_dir = os.path.join(destination_dir, 'data', patient, session, 'rr_intervals')
        os.makedirs(rr_dir, exist_ok=True)
        rr_file = os.path.join(rr_dir, f"{session}_rr_intervals.csv")

        if os.path.exists(rr_file):
            rr_data = pd.read_csv(rr_file)
            print(f"{Fore.GREEN}RR data found{Style.RESET_ALL}")
        else:
            rr_data = detect_r_peaks_and_calculate_rr(ecg_data)
            # Save RR data
            rr_data.to_csv(rr_file, index=False)

        # Initialize combined_data with RR data
        combined_data = rr_data.copy()
    else:
        print(f"{Fore.YELLOW}RR data already exists for {patient}/{session}{Style.RESET_ALL}")
        combined_data = pd.read_csv(rr_intervals_file)
        rr_data = pd.read_csv(rr_intervals_file)

    # Process simulator data and detect crashes

    crash_data_file = os.path.join('data', patient, session, 'crashes', f"{session}_crash_episodes.csv")
    print(os.path.exists(crash_data_file))

    if not os.path.exists(crash_data_file):
        print(f"{Fore.LIGHTYELLOW_EX}Processing simulator data and detecting crashes...{Style.RESET_ALL}")
        crash_data = process_simulator_data(root_dir, destination_dir, patient, session, min_minor_crash_separation=300)
    else:
        print(f"{Fore.YELLOW}Crash data already exists for {patient}/{session}{Style.RESET_ALL}")
        crash_data = pd.read_csv(crash_data_file)

    # Process PERCLOS, yawning, and posture data
    perclos_file = os.path.join(destination_dir, 'data', patient, session, 'perclos', f"{session}_perclos.csv")
    yawn_file = os.path.join(destination_dir, 'data', patient, session, 'yawning', f"{session}_yawning.csv")
    posture_file = os.path.join(destination_dir, 'data', patient, session, 'posture', f"{session}_posture.csv")



    if os.path.exists(perclos_file):
        perclos_df = pd.read_csv(perclos_file)
    else:
        perclos_df = None

    if os.path.exists(yawn_file):
        try:
            yawn_df = pd.read_csv(yawn_file)
        except:
            yawn_df = None
    else:
        yawn_df = None

    if os.path.exists(posture_file):
        posture_df = pd.read_csv(posture_file)
    else:
        posture_df = None


    print(f"{Fore.LIGHTYELLOW_EX}Processing PERCLOS, yawning, and posture data...{Style.RESET_ALL}")
    if perclos_df is None and yawn_df is None and posture_df is None:
        perclos_df, yawn_df, posture_df = process_perclos_and_yawning(root_dir, destination_dir, patient, session)


    if perclos_df is not None and not perclos_df.empty:
        perclos_df = pd.DataFrame(columns=['time', 'PERCLOS'])
        combined_data = pd.merge(combined_data, perclos_df, on='time', how='left')


    if yawn_df is not None and not yawn_df.empty:
        # Initialize yawn duration column with zeros
        yawn_df = pd.DataFrame(columns=['start_time', 'end_time', 'duration'])
        combined_data['yawn_duration'] = 0.0

        # Iterate through yawn events and update the yawn duration
        for _, yawn in yawn_df.iterrows():
            # Create boolean mask for the time range of this yawn
            mask = (combined_data['time'].between(
                yawn['start_time'],
                yawn['end_time'],
                inclusive='both'
            ))
            # Update yawn duration for matching times
            combined_data.loc[mask, 'yawn_duration'] = yawn['duration']
    else:
        print(f"{Fore.YELLOW}Warning: No yawning data available{Style.RESET_ALL}")
        combined_data['yawn_duration'] = np.nan


    if posture_df is None:
        posture_df = pd.DataFrame(columns=['time', 'angle_difference', 'pitch', 'yaw', 'roll'])

    # Process blink detection
    print(f"{Fore.LIGHTYELLOW_EX}Processing blink detection...{Style.RESET_ALL}")
    blinking_file = os.path.join("data", patient, session, "blinks", f"{session}_blinks.csv")

    if os.path.exists(blinking_file):
        blink_df = pd.read_csv(os.path.join("data", patient, session, "blinks", f"{session}_blinks.csv"))
    else:
        blink_df = process_blinks(root_dir, patient, session)

    # Merge blink data
    if blink_df is not None and not blink_df.empty:
        combined_data = pd.merge_asof(
            combined_data,
            blink_df[['start_time', 'blink_rate']].rename(columns={'start_time': 'time'}),
            on='time',
            direction='nearest'
        )
    else:
        combined_data['blink_rate'] = np.nan

    # Merge respiratory data
    if respiratory_data is not None:
        combined_data = pd.merge_asof(combined_data, respiratory_data, on='time', direction='nearest')
    else:
        combined_data['respiratory'] = np.nan

    if respiratory_rate_df is not None and not respiratory_rate_df.empty:
        combined_data = pd.merge_asof(combined_data, respiratory_rate_df, on='time', direction='nearest')
    else:
        combined_data['respiratory_rate'] = np.nan

    # Add crashes columns
    if crash_data is not None and not crash_data.empty:
        combined_data = add_crashes_columns(combined_data, crash_data)

        destination_dir = ''
        crash_slope_dir = os.path.join(destination_dir, 'data', patient, session, 'crash_slope')
        os.makedirs(crash_slope_dir, exist_ok=True)
        crash_slope_file = os.path.join(crash_slope_dir, f"{session}_crash_slope.csv")

        if os.path.exists(crash_slope_file):
            crash_slope_data = pd.read_csv(crash_slope_file)
            print(f"{Fore.GREEN}Crash slope data found{Style.RESET_ALL}")
        else:
            create_crash_derivative_dataframe(crash_data, rr_data, patient, session)
    else:
        combined_data['crash'] = 0

    # Save combined data
    data_folder = os.path.join(destination_dir, 'data', patient)
    os.makedirs(data_folder, exist_ok=True)

    combined_file = os.path.join(data_folder, f"{session}_combined_data.csv")
    if not os.path.exists(combined_file):
        combined_data.to_csv(combined_file, index=False)
    else:
        combined_data = pd.read_csv(combined_file)
        print(f"{Fore.RED}Combined data already exists for {patient}/{session}{Style.RESET_ALL}")

    return f"{patient}/{session} - Processed successfully"


def main():
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    patient_list = get_patient_list(ROOT_DIR)
    print("Patients:", patient_list)

    patient_sessions = []
    for patient in patient_list:
        session_list = get_session_list(ROOT_DIR, patient)
        for session in session_list:
            patient_sessions.append((patient, session))

    total_sessions = len(patient_sessions)

    with tqdm(total=total_sessions, desc=f"{Fore.BLUE}Processing sessions{Style.RESET_ALL}",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for patient, session in patient_sessions:
            result = process_session(ROOT_DIR, 'data', patient, session)
            pbar.update(1)
            pbar.write(f"{Fore.GREEN}{result}{Style.RESET_ALL}")

    # Train model with properly configured paths
    model, history, scaler, cv_metrics = train_crash_detection_model(
        'data',  # Changed from 'data'
        patient_sessions,
        MODEL_SAVE_PATH
    )

    # use train_with_time_shifts
    model, history, scaler, cv_metrics = train_with_time_shifts(
        'data',
        patient_sessions,
        MODEL_SAVE_PATH
    )

    # Update paths for prediction and evaluation
    predictions_dir = os.path.join('data', "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Make predictions on new data
    slope_data_path = os.path.join('data', "slope_data.csv")
    if os.path.exists(slope_data_path):
        new_slope_data = pd.read_csv(slope_data_path)
        predictions = predict_crashes(model, scaler, new_slope_data)

        # Evaluate predictions
        true_crashes_path = os.path.join('data', "true_crashes.csv")
        if os.path.exists(true_crashes_path):
            true_crashes = pd.read_csv(true_crashes_path)
            eval_metrics = evaluate_predictions(true_crashes, predictions)

            # Plot results
            plot_path = os.path.join(predictions_dir, "predictions_plot.png")
            plot_predictions(
                new_slope_data,
                predictions,
                true_crashes,
                save_path=plot_path
            )

            # Save model summary
            summary_path = os.path.join(MODEL_SAVE_PATH, "model_summary.txt")
            save_model_summary(model, summary_path)

            # Export predictions
            predictions_path = os.path.join(predictions_dir, "predictions.csv")
            export_predictions(predictions, predictions_path)

    print(f"{Fore.YELLOW}All processing completed!{Style.RESET_ALL}")


if __name__ == '__main__':
    main()