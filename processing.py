from colorama import Fore, Style
import numpy as np
import pandas as pd
from scipy import signal, interpolate
from biosppy.signals import ecg
import os

def detect_r_peaks_and_calculate_rr(ecg_data):
    ecg_data = ecg_data.sort_values('time')
    ecg_signal = ecg_data['ecg'].values
    time_diff = ecg_data['time'].diff().mean()
    if time_diff == 0:
        raise ValueError("Time difference between samples is zero")
    sampling_rate = 1 / time_diff

    print(f"{Fore.LIGHTYELLOW_EX}Sampling rate: {sampling_rate} Hz{Style.RESET_ALL}")

    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    r_peaks = out['rpeaks']

    print(f"{Fore.LIGHTYELLOW_EX}Number of R peaks: {len(r_peaks)}{Style.RESET_ALL}")

    if len(r_peaks) < 2:
        raise ValueError("Not enough R peaks detected")

    rr_intervals = np.diff(ecg_data['time'].iloc[r_peaks])
    rr_df = pd.DataFrame({
        'time': ecg_data['time'].iloc[r_peaks[1:]],
        'RR': rr_intervals
    })

    f = interpolate.interp1d(rr_df['time'], rr_df['RR'], kind='linear', fill_value='extrapolate')
    interpolated_rr = f(ecg_data['time'])

    result_df = pd.DataFrame({
        'time': ecg_data['time'],
        'RR': interpolated_rr
    })

    return result_df


def calculate_respiratory_rate(respiratory_data, window_size=30, overlap=15):

    respiratory_data = respiratory_data.sort_values('time')


    time = respiratory_data['time'].values
    # convert time in seconds float
    time = time.astype(float)


    # convert time in seconds
    time = (time - time[0]).astype(float)

    resp_signal = respiratory_data['respiratory'].values

    print(f"{Fore.LIGHTMAGENTA_EX}Respiratory data shape: {resp_signal.shape}")
    print(f"Time range: {time[0]} to {time[-1]}")
    print(f"Respiratory signal range: {np.min(resp_signal)} to {np.max(resp_signal)}{Style.RESET_ALL}")

    # Check for NaN or infinite values
    if np.isnan(resp_signal).any() or np.isinf(resp_signal).any():
        print(f"{Fore.RED}Error: Respiratory signal contains NaN or infinite values{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'respiratory_rate'])

    # Check if the signal is constant
    if np.allclose(resp_signal, resp_signal[0]):
        print(f"{Fore.YELLOW}Warning: Respiratory signal is constant{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'respiratory_rate'])

    if len(resp_signal) < 2:
        print(f"{Fore.YELLOW}Warning: Not enough data points to calculate respiratory rate{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'respiratory_rate'])

    # Calculate sampling frequency
    time_diff = np.diff(time.astype(float))
    if np.all(time_diff == 0):
        print(f"{Fore.YELLOW}Warning: All time differences are zero{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'respiratory_rate'])

    fs = 1 / np.mean(time_diff)
    print(f"{Fore.LIGHTMAGENTA_EX}Calculated sampling frequency: {fs} Hz{Style.RESET_ALL}")

    # Ensure window_size is larger than overlap
    if window_size <= overlap:
        window_size = overlap + 1
        print(f"{Fore.YELLOW}Warning: Window size adjusted to {window_size} seconds{Style.RESET_ALL}")

    window_samples = int(window_size * fs)
    overlap_samples = int(overlap * fs)
    step_samples = max(1, window_samples - overlap_samples)

    print(f"{Fore.LIGHTMAGENTA_EX}Window samples: {window_samples}, Overlap samples: {overlap_samples}, Step samples: {step_samples}{Style.RESET_ALL}")

    respiratory_rate = []
    window_times = []

    for i in range(0, len(resp_signal) - window_samples + 1, step_samples):
        window = resp_signal[i:i + window_samples]

        if len(window) < window_samples:
            print(f"{Fore.YELLOW}Warning: Skipping incomplete window at the end{Style.RESET_ALL}")
            break

        window_time = time[i + window_samples // 2]

        # Detrend the window to remove any DC offset or linear trend
        detrended_window = signal.detrend(window)

        # Apply a Hanning window to reduce spectral leakage
        windowed_data = detrended_window * signal.windows.hann(len(detrended_window))

        if np.all(windowed_data == 0):
            print(f"{Fore.YELLOW}Warning: Window contains all zeros, skipping{Style.RESET_ALL}")
            continue

        if len(windowed_data) == 0:
            print(f"{Fore.YELLOW}Warning: Empty window, skipping{Style.RESET_ALL}")
            continue

        try:
            fft = np.fft.rfft(windowed_data)
            freqs = np.fft.rfftfreq(len(windowed_data), 1 / fs)
        except Exception as e:
            print(f"{Fore.RED}Error in FFT calculation: {str(e)}{Style.RESET_ALL}")
            print(f"Windowed data shape: {windowed_data.shape}")
            print(f"Windowed data sample: {windowed_data[:10]}")
            continue

        # Consider only frequencies corresponding to 6-30 breaths per minute
        mask = (freqs >= 6 / 60) & (freqs <= 30 / 60)
        if not np.any(mask):
            print(f"{Fore.YELLOW}Warning: No frequencies in the valid range, skipping window{Style.RESET_ALL}")
            continue

        peak_idx = np.argmax(np.abs(fft[mask]))
        peak_freq = freqs[mask][peak_idx]

        rate = peak_freq * 60

        respiratory_rate.append(rate)
        window_times.append(window_time)

    if not respiratory_rate:
        print(f"{Fore.YELLOW}Warning: No valid respiratory rates calculated{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'respiratory_rate'])

    result_df = pd.DataFrame({'time': window_times, 'respiratory_rate': respiratory_rate})
    print(f"{Fore.LIGHTMAGENTA_EX}Calculated {len(result_df)} respiratory rate values{Style.RESET_ALL}")

    # print mean respiratory rate values
    print(f"{Fore.LIGHTMAGENTA_EX}Mean respiratory rate: {np.mean(result_df['respiratory_rate']):.2f} breaths per minute{Style.RESET_ALL}")
    print(f"{Fore.LIGHTMAGENTA_EX}Standard deviation of respiratory rate: {np.std(result_df['respiratory_rate']):.2f} breaths per minute{Style.RESET_ALL}")

    return result_df


def calculate_derivative(order,rr_data):
    if 'time' in rr_data.columns:
        rr_data['time'] = rr_data['time'].astype(float)

    if order == 1:
        derivative = np.diff(rr_data['RR'])
        derivative = np.insert(derivative, 0, 0)
    elif order == 2:
        derivative = np.diff(rr_data['RR'], n=2)
        derivative = np.insert(derivative, 0, 0)
        derivative = np.insert(derivative, 0, 0)
    else:
        raise ValueError("Invalid order for derivative calculation")

    return derivative


def slope_of_2nd_derivative(rr_data):
    derivative = calculate_derivative(2, rr_data)
    time = rr_data['time'].values

    # Calculate time differences
    time_diff = np.diff(time)

    # Ensure derivative and time_diff have the same length
    # The derivative already has the same length as the input due to np.insert
    # We need to handle the last point specially since we can't calculate a slope there
    derivative = derivative[:-1]  # Remove last point

    # Calculate slope
    slope = derivative / time_diff

    # Create DataFrame with the results
    result_df = pd.DataFrame({
        'time': rr_data['time'].iloc[:-1],  # Remove last point to match slope length
        'RR': rr_data['RR'].iloc[:-1],  # Remove last point to match slope length
        'slope': slope
    })

    return result_df

# create a method to merge the crash data with the slope data
def create_crash_derivative_dataframe(crash_data, rr_data, patient, session):
    if not isinstance(crash_data, pd.DataFrame) or crash_data.empty:
        print(f"{Fore.YELLOW}Warning: Invalid or empty crash data{Style.RESET_ALL}")
        return pd.DataFrame()

    # Convert crash data to episodes format if it's simulator data
    if 'Road Position (m)' in crash_data.columns:
        print(f"{Fore.CYAN}Converting simulator data to crash episodes...{Style.RESET_ALL}")
        from crashes import process_simulator_data_to_episodes
        crash_episodes = process_simulator_data_to_episodes(crash_data)
        if crash_episodes is None or crash_episodes.empty:
            print(f"{Fore.YELLOW}No crash episodes detected{Style.RESET_ALL}")
            return pd.DataFrame()
    else:
        crash_episodes = crash_data

    # Verify required columns exist
    required_columns = ['start_time', 'end_time', 'type']
    if not all(col in crash_episodes.columns for col in required_columns):
        print(
            f"{Fore.RED}Error: Missing required columns in crash episodes data. Required: {required_columns}{Style.RESET_ALL}")
        return pd.DataFrame()

    # Ensure time columns are float type
    crash_episodes = crash_episodes.copy()
    crash_episodes['start_time'] = pd.to_numeric(crash_episodes['start_time'], errors='coerce')
    crash_episodes['end_time'] = pd.to_numeric(crash_episodes['end_time'], errors='coerce')
    crash_episodes['duration'] = crash_episodes['end_time'] - crash_episodes['start_time']

    # Remove any rows with invalid time values
    crash_episodes = crash_episodes.dropna(subset=['start_time', 'end_time'])

    if crash_episodes.empty:
        print(f"{Fore.YELLOW}No valid crash episodes after processing{Style.RESET_ALL}")
        return pd.DataFrame()

    # Sort episodes by start time
    crash_episodes = crash_episodes.sort_values('start_time')

    # Ensure RR data time column is float
    rr_data = rr_data.copy()
    rr_data['time'] = pd.to_numeric(rr_data['time'], errors='coerce')
    rr_data = rr_data.dropna(subset=['time'])

    if rr_data.empty:
        print(f"{Fore.YELLOW}No valid RR interval data after processing{Style.RESET_ALL}")
        return pd.DataFrame()

    # Calculate slope
    slope_df = slope_of_2nd_derivative(rr_data)
    if slope_df.empty:
        print(f"{Fore.YELLOW}No slope data calculated{Style.RESET_ALL}")
        return pd.DataFrame()

    # Create an empty list to store results
    crash_slope_rows = []

    # Process each crash episode
    for _, crash in crash_episodes.iterrows():
        # Find all slope measurements within the crash time window
        mask = (slope_df['time'] >= crash['start_time']) & (slope_df['time'] <= crash['end_time'])
        slopes_during_crash = slope_df[mask]

        if not slopes_during_crash.empty:
            # Calculate statistics for the crash period
            crash_slope_rows.append({
                'start_time': crash['start_time'],
                'end_time': crash['end_time'],
                'duration': crash['end_time'] - crash['start_time'],
                'avg_slope': slopes_during_crash['slope'].mean(),
                'max_slope': slopes_during_crash['slope'].max(),
                'min_slope': slopes_during_crash['slope'].min(),
                'avg_rr': slopes_during_crash['RR'].mean(),
                'n_measurements': len(slopes_during_crash),
                'crash_type': crash.get('type', 'Unknown')
            })

    # Create the final dataframe
    crash_slope = pd.DataFrame(crash_slope_rows)

    if crash_slope.empty:
        print(f"{Fore.YELLOW}No slope measurements found within crash time windows{Style.RESET_ALL}")
        return pd.DataFrame()

    # Save the results
    crash_slope_dir = os.path.join('', 'data', patient, session, 'crash_slope')
    os.makedirs(crash_slope_dir, exist_ok=True)
    crash_slope_file = os.path.join(crash_slope_dir, f"{session}_crash_slope.csv")
    crash_slope.to_csv(crash_slope_file, index=False)

    print(f"{Fore.GREEN}Successfully processed {len(crash_slope)} crash episodes")
    print(f"Crash slope data saved to {crash_slope_file}{Style.RESET_ALL}")

    return crash_slope

