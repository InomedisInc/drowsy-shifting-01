import pandas as pd
import numpy as np
import os
from colorama import Fore, Style, init

init()

# Constants
VEHICLE_WIDTH = 1.676
ROAD_WIDTH = 3.3528
ROAD_POSITION_LIMITS = {
    "Left": 0 + (VEHICLE_WIDTH / 2),
    "Center": ROAD_WIDTH / 2,
    "Right": ROAD_WIDTH - (VEHICLE_WIDTH / 2),
}

def add_crashes_columns(combined_data, crash_data):
    combined_data['crash'] = 0

    if not isinstance(crash_data, pd.DataFrame):
        print(f"{Fore.YELLOW}Warning: crash_data is not a DataFrame{Style.RESET_ALL}")
        return combined_data

    if crash_data.empty:
        print(f"{Fore.YELLOW}Warning: Empty crash data{Style.RESET_ALL}")
        return combined_data

    # Check if crash_data is from simulator or processed crash episodes
    if 'Road Position (m)' in crash_data.columns:
        # Data is from simulator, need to process it first
        crash_episodes = process_simulator_data_to_episodes(crash_data)
        if crash_episodes is None or crash_episodes.empty:
            return combined_data
    else:
        # Assuming data is already in episodes format
        crash_episodes = crash_data

    # Verify required columns exist
    required_columns = ['start_time', 'end_time', 'type']
    if not all(col in crash_episodes.columns for col in required_columns):
        print(
            f"{Fore.RED}Error: Missing required columns in crash episodes data. Required: {required_columns}{Style.RESET_ALL}")
        return combined_data

    try:
        crash_episodes['start_time'] = crash_episodes['start_time'].astype(float)
        crash_episodes['end_time'] = crash_episodes['end_time'].astype(float)

        for _, crash in crash_episodes.iterrows():
            mask = (combined_data['time'] >= crash['start_time']) & (combined_data['time'] <= crash['end_time'])
            combined_data.loc[mask, 'crash'] = 2 if crash['type'] == 'Major' else 1

    except Exception as e:
        print(f"{Fore.RED}Error processing crash episodes: {str(e)}{Style.RESET_ALL}")

    return combined_data


def process_simulator_data_to_episodes(simulator_df, vehicle_proportion_outside=0.5, min_minor_crash_separation=300):

    try:
        # Verify required columns
        if 'Road Position (m)' not in simulator_df.columns or 'time' not in simulator_df.columns:
            print(f"{Fore.RED}Error: Missing required columns in simulator data{Style.RESET_ALL}")
            return None

        road_position = simulator_df["Road Position (m)"]
        time = simulator_df["time"].astype(float)

        # Detect minor crashes
        minor_crash_signal = np.where(
            (road_position < ROAD_POSITION_LIMITS["Left"]) |
            (road_position > ROAD_POSITION_LIMITS["Right"]),
            1, 0
        )

        # Detect major crashes
        major_crash_signal = np.where(
            (road_position <= (ROAD_POSITION_LIMITS["Left"] - VEHICLE_WIDTH * vehicle_proportion_outside)) |
            (road_position >= (ROAD_POSITION_LIMITS["Right"] + VEHICLE_WIDTH * vehicle_proportion_outside)),
            1, 0
        )

        # Combine crash signals (2 for major, 1 for minor)
        crash_signal = np.maximum(minor_crash_signal, 2 * major_crash_signal)

        # Detect episodes
        episode_starts = np.where(np.diff(crash_signal) != 0)[0] + 1
        if len(episode_starts) == 0:
            return pd.DataFrame(columns=['start_time', 'end_time', 'duration', 'type'])

        episode_ends = np.append(episode_starts[1:] - 1, len(crash_signal) - 1)

        # Create episodes
        episodes = []
        last_minor_crash_end = float('-inf')  # Track the end time of the last minor crash

        for start, end in zip(episode_starts, episode_ends):
            if crash_signal[start] != 0:  # If this is a crash episode
                episode_type = "Major" if crash_signal[start] == 2 else "Minor"
                start_time = float(time[start])
                end_time = float(time[end])
                duration = end_time - start_time

                if episode_type == "Major":
                    # Always include major crashes
                    episodes.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'type': episode_type
                    })
                    last_minor_crash_end = end_time  # Update last crash time for major crashes too
                else:  # Minor crash
                    # Check if enough time has passed since the last crash
                    if (start_time - last_minor_crash_end) >= min_minor_crash_separation:
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
            print(f"Minor crash separation: {min_minor_crash_separation} seconds{Style.RESET_ALL}")

        return episodes_df

    except Exception as e:
        print(f"{Fore.RED}Error processing simulator data: {str(e)}{Style.RESET_ALL}")
        return None


def process_simulator_data(root_dir, destination_dir, patient, session, vehicle_proportion_outside=0.5,
                           min_minor_crash_separation=300):
    try:
        aligned_folder = f"{session}_aligned"
        simulator_file = os.path.join(root_dir, patient, session, aligned_folder, f"{session}_simulator_data.csv")

        print(f"{Fore.MAGENTA}Processing {patient}/{session}...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Simulator file: {simulator_file}{Style.RESET_ALL}")

        if not os.path.exists(simulator_file):
            print(f"{Fore.RED}Simulator data file not found for {patient}/{session}{Style.RESET_ALL}")
            return None

        simulator_df = pd.read_csv(simulator_file)
        if simulator_df.empty:
            print(f"{Fore.RED}Empty simulator data for {patient}/{session}{Style.RESET_ALL}")
            return None

        print(f"{Fore.GREEN}Successfully read simulator data. Shape: {simulator_df.shape}{Style.RESET_ALL}")

        # Process crash episodes with minimum separation parameter
        episodes_df = process_simulator_data_to_episodes(
            simulator_df,
            vehicle_proportion_outside=vehicle_proportion_outside,
            min_minor_crash_separation=min_minor_crash_separation
        )



        if episodes_df is None or episodes_df.empty:
            print(f"{Fore.YELLOW}No crash episodes detected for {patient}/{session}{Style.RESET_ALL}")
            return None

        # Save episodes dataframe
        crash_dir = os.path.join(destination_dir, 'data', patient, session, 'crashes')
        os.makedirs(crash_dir, exist_ok=True)
        crash_episode_file = os.path.join(crash_dir, f"{session}_crash_episodes.csv")
        episodes_df.to_csv(crash_episode_file, index=False)

        assert False

        print(f"{Fore.GREEN}Crash episodes saved to {crash_episode_file}")
        print(f"Number of episodes: {len(episodes_df)}")
        print(f"Minor crashes: {sum(episodes_df['type'] == 'Minor')}")
        print(f"Major crashes: {sum(episodes_df['type'] == 'Major')}{Style.RESET_ALL}")

        return episodes_df

    except Exception as e:
        print(f"{Fore.RED}Error in process_simulator_data: {str(e)}{Style.RESET_ALL}")
        return None

