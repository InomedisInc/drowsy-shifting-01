import numpy as np
import pandas as pd
from scipy import signal
import cupy as cp
from numba import cuda, jit
import tensorflow as tf
from biosppy.signals import ecg
import os
from colorama import Fore, Style

# Initialize GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

@cuda.jit
def calculate_rr_intervals_gpu(r_peaks, times, output):
    """CUDA kernel for parallel RR interval calculation"""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)  # Total number of threads in the grid

    # Process multiple elements per thread if needed
    for i in range(idx, len(output), stride):
        if i < len(r_peaks) - 1:
            output[i] = times[r_peaks[i + 1]] - times[r_peaks[i]]


def detect_r_peaks_and_calculate_rr(ecg_data):
    """GPU-accelerated R-peak detection and RR interval calculation"""
    try:
        ecg_data = ecg_data.sort_values('time')

        # Move data to GPU
        ecg_signal = cp.asarray(ecg_data['ecg'].values)
        time_gpu = cp.asarray(ecg_data['time'].values)

        time_diff = float(cp.diff(time_gpu).mean())
        if time_diff == 0:
            raise ValueError("Time difference between samples is zero")

        sampling_rate = 1 / time_diff
        print(f"{Fore.LIGHTYELLOW_EX}Sampling rate: {sampling_rate} Hz{Style.RESET_ALL}")

        # Process ECG signal on GPU
        nyquist = sampling_rate / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, cp.asnumpy(ecg_signal))

        # Detect R peaks
        out = ecg.ecg(signal=filtered_signal, sampling_rate=sampling_rate, show=False)
        r_peaks = out['rpeaks']

        print(f"{Fore.LIGHTYELLOW_EX}Number of R peaks: {len(r_peaks)}{Style.RESET_ALL}")

        if len(r_peaks) < 2:
            raise ValueError("Not enough R peaks detected")

        # Calculate RR intervals on GPU with optimized configuration
        rr_intervals = cp.zeros(len(r_peaks) - 1)

        # Get optimal block configuration
        threads_per_block, blocks_per_grid = optimal_block_config(len(r_peaks) - 1)

        print(f"{Fore.CYAN}GPU Configuration:")
        print(f"Threads per block: {threads_per_block}")
        print(f"Blocks per grid: {blocks_per_grid}")
        print(f"Total threads: {threads_per_block * blocks_per_grid}{Style.RESET_ALL}")

        # Launch kernel with optimized configuration
        calculate_rr_intervals_gpu[blocks_per_grid, threads_per_block](
            cuda.to_device(r_peaks),
            cuda.to_device(ecg_data['time'].values),
            cuda.to_device(rr_intervals)
        )

        # Create DataFrame with results
        rr_df = pd.DataFrame({
            'time': ecg_data['time'].iloc[r_peaks[1:]],
            'RR': cp.asnumpy(rr_intervals)
        })

        # Use scipy's interp1d for interpolation
        from scipy.interpolate import interp1d
        f = interp1d(
            rr_df['time'].values,
            rr_df['RR'].values,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        # Move time values back to CPU for interpolation
        time_values = cp.asnumpy(time_gpu)
        interpolated_rr = f(time_values)

        result_df = pd.DataFrame({
            'time': ecg_data['time'],
            'RR': interpolated_rr
        })

        print(f"{Fore.GREEN}RR interval calculation completed successfully{Style.RESET_ALL}")
        return result_df

    except Exception as e:
        print(f"{Fore.RED}Error in R-peak detection: {str(e)}{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'RR'])


@cuda.jit
def process_respiratory_window_gpu(window, fs, hann_window, output):
    """CUDA kernel for processing respiratory windows"""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Process multiple elements per thread if needed
    for i in range(idx, window.shape[0], stride):
        if i < window.shape[0]:
            # Calculate window mean for detrending
            mean_val = 0
            for j in range(window.shape[0]):
                mean_val += window[j]
            mean_val /= window.shape[0]

            # Apply pre-calculated Hanning window
            detrended = window[i] - mean_val
            output[i] = detrended * hann_window[i]


def optimal_block_config(data_size):
    """Calculate optimal block and grid configuration for GPU"""
    # CUDA architectural constraints
    warp_size = 32
    min_blocks_per_sm = 2  # Minimum blocks per streaming multiprocessor
    max_threads_per_block = 1024  # Increased from 512 to allow more threads
    min_warps_per_block = 2  # Minimum warps per block for good occupancy

    # Calculate minimum number of threads needed per block
    min_threads = warp_size * min_warps_per_block

    # Calculate optimal threads per block
    threads_per_block = min(max_threads_per_block,
                            max(min_threads,
                                ((data_size + 255) // 256) * warp_size))

    # Ensure threads_per_block is multiple of warp_size
    threads_per_block = ((threads_per_block + warp_size - 1) // warp_size) * warp_size

    # Calculate number of blocks needed
    # Increase minimum number of blocks to improve occupancy
    min_blocks = min_blocks_per_sm * 8  # Assuming at least 8 SMs on modern GPUs
    blocks_per_grid = max(min_blocks,
                          (data_size + threads_per_block - 1) // threads_per_block)

    print(f"{Fore.CYAN}GPU Configuration Details:")
    print(f"Data size: {data_size}")
    print(f"Threads per block: {threads_per_block}")
    print(f"Blocks per grid: {blocks_per_grid}")
    print(f"Total threads: {threads_per_block * blocks_per_grid}")
    print(f"Warps per block: {threads_per_block // warp_size}")
    print(f"Occupancy ratio: {(threads_per_block * blocks_per_grid) / data_size:.2f}{Style.RESET_ALL}")

    return threads_per_block, blocks_per_grid


@cuda.jit
def process_respiratory_window_gpu(window, fs, hann_window, output):
    """CUDA kernel for processing respiratory windows with improved thread utilization"""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Each thread processes multiple elements to improve utilization
    for i in range(idx, window.shape[0], stride):
        if i < window.shape[0]:
            # Calculate local mean for detrending
            local_mean = 0.0
            start_idx = max(0, i - 16)
            end_idx = min(window.shape[0], i + 17)
            count = 0

            for j in range(start_idx, end_idx):
                local_mean += window[j]
                count += 1

            local_mean /= count

            # Apply pre-calculated Hanning window with local detrending
            output[i] = (window[i] - local_mean) * hann_window[i]


def calculate_respiratory_rate(respiratory_data, window_size=30, overlap=15):
    """Calculate respiratory rate using optimized GPU acceleration"""
    try:
        # Move data to GPU
        resp_signal = cp.asarray(respiratory_data['respiratory'].values)
        time = cp.asarray(respiratory_data['time'].values.astype(float))

        # Calculate sampling frequency
        time_diff = cp.diff(time)
        if cp.all(time_diff == 0):
            print(f"{Fore.YELLOW}Warning: All time differences are zero{Style.RESET_ALL}")
            return pd.DataFrame(columns=['time', 'respiratory_rate'])

        fs = 1 / float(cp.mean(time_diff))
        print(f"{Fore.LIGHTMAGENTA_EX}Sampling frequency: {fs} Hz{Style.RESET_ALL}")

        # Calculate window parameters
        window_samples = int(window_size * fs)
        overlap_samples = int(overlap * fs)
        step_samples = max(1, window_samples - overlap_samples)

        # Pre-calculate Hanning window
        hann_window = np.hanning(window_samples)
        hann_window_gpu = cuda.to_device(hann_window)

        # Get optimal block configuration
        threads_per_block, blocks_per_grid = optimal_block_config(window_samples)

        # Process windows in batches for better GPU utilization
        batch_size = 10  # Process multiple windows simultaneously
        respiratory_rate = []
        window_times = []

        for i in range(0, len(resp_signal) - window_samples + 1, step_samples * batch_size):
            batch_windows = []
            batch_times = []

            # Prepare batch of windows
            for j in range(batch_size):
                if i + j * step_samples >= len(resp_signal) - window_samples + 1:
                    break

                window = resp_signal[i + j * step_samples:i + j * step_samples + window_samples]
                window_time = time[i + j * step_samples + window_samples // 2]

                batch_windows.append(window)
                batch_times.append(window_time)

            if not batch_windows:
                break

            # Process batch on GPU
            batch_windows_array = cp.array(batch_windows)
            output_gpu = cp.zeros_like(batch_windows_array)

            # Launch kernel for each window in batch
            for idx, window in enumerate(batch_windows_array):
                process_respiratory_window_gpu[blocks_per_grid, threads_per_block](
                    cuda.to_device(window),
                    fs,
                    hann_window_gpu,
                    cuda.to_device(output_gpu[idx])
                )

                # Calculate FFT on GPU
                fft = cp.fft.rfft(output_gpu[idx])
                freqs = cp.fft.rfftfreq(len(output_gpu[idx]), 1 / fs)

                # Find peak in respiratory range
                mask = (freqs >= 6 / 60) & (freqs <= 30 / 60)
                if cp.any(mask):
                    peak_idx = cp.argmax(cp.abs(fft[mask]))
                    peak_freq = float(freqs[mask][peak_idx])
                    rate = peak_freq * 60
                    respiratory_rate.append(rate)
                    window_times.append(float(batch_times[idx]))

        if not respiratory_rate:
            print(f"{Fore.YELLOW}Warning: No valid respiratory rates calculated{Style.RESET_ALL}")
            return pd.DataFrame(columns=['time', 'respiratory_rate'])

        result_df = pd.DataFrame({
            'time': window_times,
            'respiratory_rate': respiratory_rate
        })

        print(f"{Fore.GREEN}Respiratory rate calculation completed")
        print(f"Mean rate: {np.mean(result_df['respiratory_rate']):.2f} breaths/min")
        print(f"Std dev: {np.std(result_df['respiratory_rate']):.2f} breaths/min{Style.RESET_ALL}")

        return result_df

    except Exception as e:
        print(f"{Fore.RED}Error in respiratory rate calculation: {str(e)}{Style.RESET_ALL}")
        return pd.DataFrame(columns=['time', 'respiratory_rate'])

@cuda.jit
def calculate_derivative_gpu(signal, output, order):
    """CUDA kernel for derivative calculation"""
    idx = cuda.grid(1)
    if order == 1:
        if idx < output.shape[0]:
            output[idx] = signal[idx + 1] - signal[idx]
    elif order == 2:
        if idx < output.shape[0]:
            output[idx] = signal[idx + 2] - 2 * signal[idx + 1] + signal[idx]


def calculate_derivative(order, rr_data):
    """Calculate derivatives using GPU acceleration"""
    try:
        # Move data to GPU
        rr_signal = cp.asarray(rr_data['RR'].values)

        # Prepare output array with correct shape
        if order == 1:
            derivative = cp.zeros(len(rr_signal) - 1)
        elif order == 2:
            derivative = cp.zeros(len(rr_signal) - 2)
        else:
            raise ValueError("Invalid order for derivative calculation")

        # Configure CUDA kernel
        threadsperblock = 256
        blockspergrid = (len(derivative) + (threadsperblock - 1)) // threadsperblock

        # Calculate derivative on GPU
        calculate_derivative_gpu[blockspergrid, threadsperblock](
            rr_signal, derivative, order
        )

        # Move result back to CPU
        result = cp.asnumpy(derivative)

        # Pad result to match original length
        if order == 1:
            result = np.pad(result, (0, 1), mode='edge')
        elif order == 2:
            result = np.pad(result, (0, 2), mode='edge')

        return result

    except Exception as e:
        print(f"{Fore.RED}Error calculating derivative: {str(e)}{Style.RESET_ALL}")
        return None


def slope_of_2nd_derivative(rr_data):
    """Calculate slope of second derivative using GPU acceleration"""
    try:
        derivative = calculate_derivative(2, rr_data)
        if derivative is None:
            return pd.DataFrame()

        # Ensure time and derivative arrays have the same length
        time = rr_data['time'].values
        derivative = derivative[:len(time)-1]  # Remove last element to match diff length
        time_diff = np.diff(time)  # Calculate time differences

        if len(derivative) != len(time_diff):
            print(f"{Fore.YELLOW}Warning: Length mismatch. Adjusting arrays.{Style.RESET_ALL}")
            min_len = min(len(derivative), len(time_diff))
            derivative = derivative[:min_len]
            time_diff = time_diff[:min_len]

        # Calculate slope
        slope = derivative / time_diff

        # Create results DataFrame, ensuring all arrays have the same length
        result_df = pd.DataFrame({
            'time': rr_data['time'].iloc[:len(slope)],
            'RR': rr_data['RR'].iloc[:len(slope)],
            'slope': slope
        })

        return result_df

    except Exception as e:
        print(f"{Fore.RED}Error calculating slope: {str(e)}{Style.RESET_ALL}")
        print(f"Shapes - derivative: {derivative.shape if derivative is not None else 'None'}, "
              f"time_diff: {time_diff.shape if 'time_diff' in locals() else 'Not calculated'}")
        return pd.DataFrame()


def create_crash_derivative_dataframe(crash_data, rr_data, patient, session):
    """Create crash derivative dataframe with GPU acceleration"""
    try:
        if not isinstance(crash_data, pd.DataFrame) or crash_data.empty:
            print(f"{Fore.YELLOW}Warning: Invalid or empty crash data{Style.RESET_ALL}")
            return pd.DataFrame()

        # Convert crash data to episodes format if needed
        if 'Road Position (m)' in crash_data.columns:
            print(f"{Fore.CYAN}Converting simulator data to crash episodes...{Style.RESET_ALL}")
            from crashes import process_simulator_data_to_episodes
            crash_episodes = process_simulator_data_to_episodes(crash_data)
            if crash_episodes is None or crash_episodes.empty:
                return pd.DataFrame()
        else:
            crash_episodes = crash_data

        # Verify required columns
        required_columns = ['start_time', 'end_time', 'type']
        if not all(col in crash_episodes.columns for col in required_columns):
            print(f"{Fore.RED}Error: Missing required columns{Style.RESET_ALL}")
            return pd.DataFrame()

        # Process crash episodes
        crash_episodes = crash_episodes.copy()
        for col in ['start_time', 'end_time']:
            crash_episodes[col] = pd.to_numeric(crash_episodes[col], errors='coerce')
        crash_episodes['duration'] = crash_episodes['end_time'] - crash_episodes['start_time']
        crash_episodes = crash_episodes.dropna(subset=['start_time', 'end_time'])

        if crash_episodes.empty:
            print(f"{Fore.YELLOW}No valid crash episodes after processing{Style.RESET_ALL}")
            return pd.DataFrame()

        # Calculate slope with appropriate length checking
        slope_df = slope_of_2nd_derivative(rr_data)
        if slope_df.empty:
            print(f"{Fore.YELLOW}No slope data calculated{Style.RESET_ALL}")
            return pd.DataFrame()

        # Process each crash episode
        crash_slope_rows = []
        for _, crash in crash_episodes.iterrows():
            mask = (slope_df['time'] >= crash['start_time']) & (slope_df['time'] <= crash['end_time'])
            slopes_during_crash = slope_df[mask]

            if not slopes_during_crash.empty:
                crash_slope_rows.append({
                    'start_time': crash['start_time'],
                    'end_time': crash['end_time'],
                    'duration': crash['duration'],
                    'avg_slope': slopes_during_crash['slope'].mean(),
                    'max_slope': slopes_during_crash['slope'].max(),
                    'min_slope': slopes_during_crash['slope'].min(),
                    'avg_rr': slopes_during_crash['RR'].mean(),
                    'n_measurements': len(slopes_during_crash),
                    'crash_type': crash.get('type', 'Unknown')
                })

        # Create final DataFrame
        crash_slope = pd.DataFrame(crash_slope_rows)
        if not crash_slope.empty:
            # Save results
            crash_slope_dir = os.path.join('data', patient, session, 'crash_slope')
            os.makedirs(crash_slope_dir, exist_ok=True)
            crash_slope_file = os.path.join(crash_slope_dir, f"{session}_crash_slope.csv")
            crash_slope.to_csv(crash_slope_file, index=False)

            print(f"{Fore.GREEN}Successfully processed {len(crash_slope)} crash episodes")
            print(f"Data saved to {crash_slope_file}{Style.RESET_ALL}")

        return crash_slope

    except Exception as e:
        print(f"{Fore.RED}Error creating crash derivative dataframe: {str(e)}{Style.RESET_ALL}")
        return pd.DataFrame()