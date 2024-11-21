import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cupy as cp
from tqdm import tqdm
from colorama import Fore, Style, init
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize colorama and GPU configuration
init()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{Fore.GREEN}GPU detected and configured successfully{Style.RESET_ALL}")
    except RuntimeError as e:
        print(f"{Fore.RED}GPU configuration error: {str(e)}{Style.RESET_ALL}")

# Constants
ROOT_DIR = "F:\\Data_Driving\\Recordings"
DESTINATION_DIR = ""  # Specify the destination directory
PERCLOS_FILENAME = "perclos_data"
ECG_FILENAME = "biopac"
MAX_WORKERS = 4  # Adjust based on your system

# Import GPU-optimized modules
from gpu_perclos import process_perclos_and_yawning, process_blinks
from gpu_processing import (
    detect_r_peaks_and_calculate_rr,
    calculate_respiratory_rate,
    create_crash_derivative_dataframe
)
from gpu_crashes import (
    add_crashes_columns,
    process_simulator_data
)
from gpu_models import (
    train_crash_detection_model,
    predict_crashes,
    evaluate_predictions,
    save_model_summary,
    export_predictions,
    plot_predictions
)


def get_patient_list(root_dir):
    """Get list of patients"""
    return sorted([f for f in os.listdir(root_dir)
                   if f.startswith('P0') and not f.endswith('lnk')])


def get_session_list(root_dir, patient):
    """Get list of sessions for a patient"""
    return [f for f in os.listdir(os.path.join(root_dir, patient))
            if f.startswith(f'{patient}_S0')]


def process_session(root_dir, destination_dir, patient, session):
    """Process a single session with GPU acceleration"""
    try:
        aligned_folder = f"{session}_aligned"
        ecg_file = os.path.join(root_dir, patient, session, aligned_folder,
                                f"{session}_{ECG_FILENAME}.csv")

        print(f"{Fore.MAGENTA}Processing {patient}/{session}...{Style.RESET_ALL}")
        print(f"{Fore.LIGHTYELLOW_EX}ECG file: {ecg_file}{Style.RESET_ALL}")

        # Read ECG data
        ecg_data = pd.read_csv(ecg_file, engine='python')
        ecg_data['time'] = ecg_data['time'].astype(float)

        # Process respiratory data
        respiratory_data = None
        respiratory_rate_df = None
        if 'Biopac_0' in ecg_data.columns:
            respiratory_data = ecg_data[['time', 'Biopac_0']].rename(
                columns={'Biopac_0': 'respiratory'})

            resp_rate_dir = os.path.join(destination_dir, 'data', patient, session,
                                         'respiratory_rate')
            os.makedirs(resp_rate_dir, exist_ok=True)
            resp_rate_file = os.path.join(resp_rate_dir,
                                          f"{session}_respiratory_rate.csv")

            if os.path.exists(resp_rate_file):
                respiratory_rate_df = pd.read_csv(resp_rate_file)
                print(f"{Fore.GREEN}Respiratory rate data found{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Processing respiratory data...{Style.RESET_ALL}")
                respiratory_rate_df = calculate_respiratory_rate(respiratory_data)
                if not respiratory_rate_df.empty:
                    respiratory_rate_df.to_csv(resp_rate_file, index=False)

        # Process ECG data with GPU acceleration
        ecg_data = ecg_data[['time', 'Biopac_2']].rename(columns={'Biopac_2': 'ecg'})
        rr_dir = os.path.join(destination_dir, 'data', patient, session, 'rr_intervals')
        os.makedirs(rr_dir, exist_ok=True)
        rr_file = os.path.join(rr_dir, f"{session}_rr_intervals.csv")

        if os.path.exists(rr_file):
            rr_data = pd.read_csv(rr_file)
            print(f"{Fore.GREEN}RR data found{Style.RESET_ALL}")
        else:
            rr_data = detect_r_peaks_and_calculate_rr(ecg_data)
            rr_data.to_csv(rr_file, index=False)

        # Initialize combined_data with RR data
        combined_data = rr_data.copy()

        # Process simulator data and detect crashes
        simulator_file = os.path.join(root_dir, patient, session, aligned_folder,
                                      f"{session}_simulator_data.csv")
        if not os.path.exists(simulator_file):
            print(f"{Fore.LIGHTYELLOW_EX}Processing simulator data...{Style.RESET_ALL}")
            crash_data = process_simulator_data(root_dir, destination_dir, patient,
                                                session, min_minor_crash_separation=300)
        else:
            crash_data = pd.read_csv(simulator_file)

        # Process PERCLOS, yawning, and posture data
        print(f"{Fore.LIGHTYELLOW_EX}Processing PERCLOS and yawning...{Style.RESET_ALL}")
        perclos_df, yawn_df, posture_df = process_perclos_and_yawning(
            root_dir, destination_dir, patient, session)

        # Merge PERCLOS data
        if perclos_df is not None and not perclos_df.empty:
            combined_data = pd.merge(combined_data, perclos_df, on='time', how='left')

        # Merge yawning data
        if yawn_df is not None and not yawn_df.empty:
            combined_data['yawn_duration'] = 0.0
            for _, yawn in yawn_df.iterrows():
                mask = (combined_data['time'].between(
                    yawn['start_time'],
                    yawn['end_time'],
                    inclusive='both'
                ))
                combined_data.loc[mask, 'yawn_duration'] = yawn['duration']
        else:
            combined_data['yawn_duration'] = np.nan

        # Process blink detection
        print(f"{Fore.LIGHTYELLOW_EX}Processing blink detection...{Style.RESET_ALL}")
        blinking_file = os.path.join(destination_dir, "data", patient, session,
                                     "blinks", f"{session}_blinks.csv")

        if os.path.exists(blinking_file):
            blink_df = pd.read_csv(blinking_file)
        else:
            blink_df = process_blinks(root_dir, patient, session)

        # Merge blink data
        if blink_df is not None and not blink_df.empty:
            combined_data = pd.merge_asof(
                combined_data,
                blink_df[['start_time', 'blink_rate']].rename(
                    columns={'start_time': 'time'}),
                on='time',
                direction='nearest'
            )
        else:
            combined_data['blink_rate'] = np.nan

        # Merge respiratory data
        if respiratory_data is not None:
            combined_data = pd.merge_asof(
                combined_data,
                respiratory_data,
                on='time',
                direction='nearest'
            )
        else:
            combined_data['respiratory'] = np.nan

        if respiratory_rate_df is not None and not respiratory_rate_df.empty:
            combined_data = pd.merge_asof(
                combined_data,
                respiratory_rate_df,
                on='time',
                direction='nearest'
            )
        else:
            combined_data['respiratory_rate'] = np.nan

        # Add crashes columns
        if crash_data is not None and not crash_data.empty:
            combined_data = add_crashes_columns(combined_data, crash_data)

            crash_slope_dir = os.path.join(destination_dir, 'data', patient,
                                           session, 'crash_slope')
            os.makedirs(crash_slope_dir, exist_ok=True)
            crash_slope_file = os.path.join(crash_slope_dir,
                                            f"{session}_crash_slope.csv")

            if not os.path.exists(crash_slope_file):
                create_crash_derivative_dataframe(crash_data, rr_data, patient, session)
        else:
            combined_data['crash'] = 0

        # Save combined data
        data_folder = os.path.join(destination_dir, 'data', patient)
        os.makedirs(data_folder, exist_ok=True)
        combined_file = os.path.join(data_folder, f"{session}_combined_data.csv")

        if not os.path.exists(combined_file):
            combined_data.to_csv(combined_file, index=False)
            print(f"{Fore.GREEN}Combined data saved to {combined_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Combined data already exists{Style.RESET_ALL}")

        return f"{patient}/{session} - Processed successfully"

    except Exception as e:
        print(f"{Fore.RED}Error processing {patient}/{session}: {str(e)}{Style.RESET_ALL}")
        return f"{patient}/{session} - Failed: {str(e)}"


def process_sessions_parallel(root_dir, destination_dir, patient_sessions):
    """Process sessions in parallel using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for patient, session in patient_sessions:
            future = executor.submit(
                process_session,
                root_dir,
                destination_dir,
                patient,
                session
            )
            futures.append(future)

        with tqdm(total=len(futures),
                  desc=f"{Fore.BLUE}Processing sessions{Style.RESET_ALL}",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                pbar.write(f"{Fore.GREEN}{result}{Style.RESET_ALL}")


def main():
    """Main function to orchestrate the entire processing pipeline"""
    try:
        # Get patient list
        patient_list = get_patient_list(ROOT_DIR)
        print(f"Patients: {patient_list}")

        # Create patient_sessions list
        patient_sessions = []
        for patient in patient_list:
            session_list = get_session_list(ROOT_DIR, patient)
            for session in session_list:
                patient_sessions.append((patient, session))

        # Process sessions in parallel
        process_sessions_parallel(ROOT_DIR, DESTINATION_DIR, patient_sessions)

        # Train model
        model_save_path = 'models'
        print(f"{Fore.CYAN}Training crash detection model...{Style.RESET_ALL}")

        model, history, scaler, cv_metrics = train_crash_detection_model(
            ROOT_DIR,
            patient_sessions,
            model_save_path
        )

        if model is not None:
            # Make predictions on new data
            new_slope_data = pd.read_csv('path/to/new/slope_data.csv')
            predictions = predict_crashes(model, scaler, new_slope_data)

            # Evaluate predictions
            true_crashes = pd.read_csv('path/to/true/crashes.csv')
            eval_metrics = evaluate_predictions(true_crashes, predictions)

            # Plot results
            plot_predictions(
                new_slope_data,
                predictions,
                true_crashes,
                save_path='path/to/save/plot.png'
            )

            # Save model summary
            save_model_summary(model, 'path/to/save/model_summary.txt')

            # Export predictions
            export_predictions(predictions, 'path/to/save/predictions.csv')

        print(f"{Fore.YELLOW}All processing completed!{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error in main execution: {str(e)}{Style.RESET_ALL}")


if __name__ == '__main__':
    main()