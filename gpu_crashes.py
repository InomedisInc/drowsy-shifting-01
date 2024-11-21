import pandas as pd
import numpy as np
import os
from colorama import Fore, Style, init
import cupy as cp
from numba import cuda
import tensorflow as tf

init()

# Constants
@cuda.jit
def detect_crashes_kernel(road_positions, vehicle_proportion, left_limit, right_limit, output_minor, output_major):
    """CUDA kernel for parallel crash detection with explicit limits"""
    idx = cuda.grid(1)
    if idx < road_positions.shape[0]:
        # Detect minor crashes
        if (road_positions[idx] < left_limit or
                road_positions[idx] > right_limit):
            output_minor[idx] = 1

        # Detect major crashes
        major_left = left_limit - (vehicle_proportion * VEHICLE_WIDTH)
        major_right = right_limit + (vehicle_proportion * VEHICLE_WIDTH)
        if (road_positions[idx] <= major_left or
                road_positions[idx] >= major_right):
            output_major[idx] = 1


def add_crashes_columns(combined_data, crash_data):
    """Add crash columns using GPU acceleration"""
    if not isinstance(crash_data, pd.DataFrame):
        print(f"{Fore.YELLOW}Warning: crash_data is not a DataFrame{Style.RESET_ALL}")
        return combined_data

    if crash_data.empty:
        print(f"{Fore.YELLOW}Warning: Empty crash data{Style.RESET_ALL}")
        return combined_data

    try:
        # Move data to GPU
        time_gpu = cp.asarray(combined_data['time'].values)
        combined_data['crash'] = 0

        if 'Road Position (m)' in crash_data.columns:
            crash_episodes = process_simulator_data_to_episodes(crash_data)
            if crash_episodes is None or crash_episodes.empty:
                return combined_data
        else:
            crash_episodes = crash_data

        required_columns = ['start_time', 'end_time', 'type']
        if not all(col in crash_episodes.columns for col in required_columns):
            print(f"{Fore.RED}Error: Missing required columns{Style.RESET_ALL}")
            return combined_data

        crash_episodes['start_time'] = crash_episodes['start_time'].astype(float)
        crash_episodes['end_time'] = crash_episodes['end_time'].astype(float)

        # Process crashes in parallel on GPU
        for _, crash in crash_episodes.iterrows():
            mask = (time_gpu >= crash['start_time']) & (time_gpu <= crash['end_time'])
            crash_value = 2 if crash['type'] == 'Major' else 1
            combined_data.loc[cp.asnumpy(mask), 'crash'] = crash_value

        return combined_data

    except Exception as e:
        print(f"{Fore.RED}Error processing crash episodes: {str(e)}{Style.RESET_ALL}")
        return combined_data


def process_simulator_data_to_episodes(simulator_df, vehicle_proportion_outside=0.5, min_minor_crash_separation=300):
    """Process simulator data using GPU acceleration"""
    try:
        if 'Road Position (m)' not in simulator_df.columns or 'time' not in simulator_df.columns:
            print(f"{Fore.RED}Error: Missing required columns{Style.RESET_ALL}")
            return None

        # Move data to GPU
        road_position = cp.asarray(simulator_df["Road Position (m)"].values)
        time = cp.asarray(simulator_df["time"].values.astype(float))

        # Prepare output arrays on GPU
        minor_crashes = cp.zeros_like(road_position, dtype=cp.int32)
        major_crashes = cp.zeros_like(road_position, dtype=cp.int32)

        # Configure CUDA kernel
        threadsperblock = 256
        blockspergrid = (road_position.shape[0] + (threadsperblock - 1)) // threadsperblock

        # Launch kernel with explicit limits
        detect_crashes_kernel[blockspergrid, threadsperblock](
            road_position,
            vehicle_proportion_outside,
            LEFT_LIMIT,
            RIGHT_LIMIT,
            minor_crashes,
            major_crashes
        )

        # Combine crash signals
        crash_signal = cp.maximum(minor_crashes, 2 * major_crashes)
        crash_signal = cp.asnumpy(crash_signal)
        time = cp.asnumpy(time)

        # Detect episodes
        episode_starts = np.where(np.diff(crash_signal) != 0)[0] + 1
        if len(episode_starts) == 0:
            return pd.DataFrame(columns=['start_time', 'end_time', 'duration', 'type'])

        episode_ends = np.append(episode_starts[1:] - 1, len(crash_signal) - 1)

        # Process episodes
        episodes = []
        last_minor_crash_end = float('-inf')

        for start, end in zip(episode_starts, episode_ends):
            if crash_signal[start] != 0:
                episode_type = "Major" if crash_signal[start] == 2 else "Minor"
                start_time = float(time[start])
                end_time = float(time[end])
                duration = end_time - start_time

                if episode_type == "Major" or (start_time - last_minor_crash_end) >= min_minor_crash_separation:
                    episodes.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'type': episode_type
                    })
                    last_minor_crash_end = end_time

        episodes_df = pd.DataFrame(episodes)

        if not episodes_df.empty:
            print(f"{Fore.GREEN}Detected crashes:")
            print(f"Total crashes: {len(episodes_df)}")
            print(f"Major crashes: {sum(episodes_df['type'] == 'Major')}")
            print(f"Minor crashes: {sum(episodes_df['type'] == 'Minor')}")
            print(f"Minor crash separation: {min_minor_crash_separation}s{Style.RESET_ALL}")

        return episodes_df

    except Exception as e:
        print(f"{Fore.RED}Error in GPU crash detection: {str(e)}{Style.RESET_ALL}")
        return None


def process_simulator_data(root_dir, destination_dir, patient, session, vehicle_proportion_outside=0.5,
                           min_minor_crash_separation=300):
    """Main function to process simulator data"""
    try:
        aligned_folder = f"{session}_aligned"
        simulator_file = os.path.join(root_dir, patient, session, aligned_folder, f"{session}_simulator_data.csv")

        print(f"{Fore.MAGENTA}Processing {patient}/{session}...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Simulator file: {simulator_file}{Style.RESET_ALL}")

        if not os.path.exists(simulator_file):
            print(f"{Fore.RED}Simulator data file not found{Style.RESET_ALL}")
            return None

        simulator_df = pd.read_csv(simulator_file)
        if simulator_df.empty:
            print(f"{Fore.RED}Empty simulator data{Style.RESET_ALL}")
            return None

        print(f"{Fore.GREEN}Successfully read simulator data. Shape: {simulator_df.shape}{Style.RESET_ALL}")

        episodes_df = process_simulator_data_to_episodes(
            simulator_df,
            vehicle_proportion_outside=vehicle_proportion_outside,
            min_minor_crash_separation=min_minor_crash_separation
        )

        if episodes_df is None or episodes_df.empty:
            print(f"{Fore.YELLOW}No crash episodes detected{Style.RESET_ALL}")
            return None

        # Save episodes dataframe
        crash_dir = os.path.join(destination_dir, 'data', patient, session, 'crashes')
        os.makedirs(crash_dir, exist_ok=True)
        crash_episode_file = os.path.join(crash_dir, f"{session}_crash_episodes.csv")
        episodes_df.to_csv(crash_episode_file, index=False)

        print(f"{Fore.GREEN}Crash episodes saved to {crash_episode_file}")
        print(f"Number of episodes: {len(episodes_df)}")
        print(f"Minor crashes: {sum(episodes_df['type'] == 'Minor')}")
        print(f"Major crashes: {sum(episodes_df['type'] == 'Major')}{Style.RESET_ALL}")

        return episodes_df

    except Exception as e:
        print(f"{Fore.RED}Error in process_simulator_data: {str(e)}{Style.RESET_ALL}")
        return None