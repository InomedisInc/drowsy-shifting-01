import cv2
import pandas as pd
import numpy as np
import os
import cupy as cp
from numba import cuda, jit
import tensorflow as tf
import urllib.request
import dlib
from datetime import datetime
from imutils import face_utils
from colorama import Fore, Style, init
from tqdm import tqdm

# Initialize colorama and GPU configuration
init()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Constants for face detection
MODEL_POINTS = cp.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])

POSTURE_THRESHOLD = 30


@cuda.jit
def calculate_distance_gpu(x1, y1, x2, y2, output):
    """CUDA kernel for calculating Euclidean distance without sqrt"""
    idx = cuda.grid(1)
    if idx < output.shape[0]:
        # Return squared distance (no sqrt needed)
        output[idx] = (x2[idx] - x1[idx])**2 + (y2[idx] - y1[idx])**2


@cuda.jit
def calculate_ear_gpu(eye_points, output):
    """CUDA kernel for calculating EAR using squared distances"""
    idx = cuda.grid(1)
    if idx < eye_points.shape[0]:
        # Calculate vertical distances (squared)
        A_squared = ((eye_points[idx, 1, 0] - eye_points[idx, 5, 0]) ** 2 +
                     (eye_points[idx, 1, 1] - eye_points[idx, 5, 1]) ** 2)
        B_squared = ((eye_points[idx, 2, 0] - eye_points[idx, 4, 0]) ** 2 +
                     (eye_points[idx, 2, 1] - eye_points[idx, 4, 1]) ** 2)

        # Calculate horizontal distance (squared)
        C_squared = ((eye_points[idx, 0, 0] - eye_points[idx, 3, 0]) ** 2 +
                     (eye_points[idx, 0, 1] - eye_points[idx, 3, 1]) ** 2)

        if C_squared > 0:
            # Using squared distances instead of actual distances
            # This maintains the relative proportions without needing sqrt
            output[idx] = (A_squared + B_squared) / (2.0 * C_squared)
        else:
            output[idx] = 0.0


def detect_blinks_gpu(shape, ear_threshold=0.25):
    """Detect blinks using GPU acceleration"""
    try:
        # Extract eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Combine both eyes into a single array
        eye_points = np.stack([left_eye, right_eye])

        # Move data to GPU
        eye_points_gpu = cuda.to_device(eye_points.astype(np.float32))
        ear_output = cuda.to_device(np.zeros(2, dtype=np.float32))

        # Configure CUDA kernel
        threadsperblock = 256
        blockspergrid = (2 + threads_per_block - 1) // threadsperblock

        # Calculate EAR
        calculate_ear_gpu[blockspergrid, threadsperblock](eye_points_gpu, ear_output)

        # Get results back from GPU
        ear_values = ear_output.copy_to_host()

        # Take square root of the result here on CPU if needed
        ear_values = np.sqrt(ear_values)
        ear = float(np.mean(ear_values))

        return ear < ear_threshold, ear

    except Exception as e:
        print(f"{Fore.RED}Error in blink detection: {str(e)}{Style.RESET_ALL}")
        return False, 0.0


def detect_perclos_and_yawning(video_path, output_dir, patient, session, start_time_offset=300,
                               duration=180, ear_threshold=0.25, mar_threshold=0.6,
                               yawn_frames=20, consecutive_frames=3):
    """Main function for PERCLOS and yawning detection using GPU acceleration"""
    try:
        # Initialize detector and predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}Error: Could not open video: {video_path}{Style.RESET_ALL}")
            return None, None, None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            print(f"{Fore.RED}Invalid video properties{Style.RESET_ALL}")
            return None, None, None

        # Calculate frame positions
        start_frame = int(start_time_offset * fps)
        frames_to_process = min(int(duration * fps), total_frames - start_frame)

        # Skip to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize tracking variables
        closed_eyes_frames = 0
        total_frames_processed = 0
        is_yawning = False
        yawn_frames_count = 0
        yawn_start_time = None

        # Initialize event lists
        yawn_events = []
        perclos_measurements = []

        # Process frames in batches for better GPU utilization
        batch_size = 32
        frames_batch = []
        times_batch = []

        pbar = tqdm(total=frames_to_process,
                    desc=f"Processing {patient}/{session}",
                    bar_format='{l_bar}{bar:30}{r_bar}')

        while total_frames_processed < frames_to_process:
            # Read batch of frames
            batch_count = 0
            frames_batch.clear()
            times_batch.clear()

            while batch_count < batch_size and total_frames_processed < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frames_batch.append(frame)
                times_batch.append((start_frame + total_frames_processed) / fps)
                batch_count += 1
                total_frames_processed += 1

            if not frames_batch:
                break

            # Process batch
            for frame, current_time in zip(frames_batch, times_batch):
                # Detect faces
                faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                for face in faces:
                    shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face)
                    shape = face_utils.shape_to_np(shape)

                    # Detect blinks
                    is_blinking, ear = detect_blinks_gpu(shape, ear_threshold)

                    if is_blinking:
                        closed_eyes_frames += 1
                    else:
                        if closed_eyes_frames >= consecutive_frames:
                            perclos_measurements.append({
                                'time': current_time,
                                'duration': closed_eyes_frames / fps
                            })
                        closed_eyes_frames = 0

                    # Process yawning
                    mouth = shape[48:68]
                    mar = calculate_mar(mouth)

                    if mar > mar_threshold:
                        yawn_frames_count += 1
                        if yawn_frames_count >= yawn_frames and not is_yawning:
                            yawn_start_time = current_time
                            is_yawning = True
                    else:
                        if is_yawning:
                            yawn_events.append({
                                'start_time': yawn_start_time,
                                'end_time': current_time,
                                'duration': current_time - yawn_start_time
                            })
                        is_yawning = False
                        yawn_frames_count = 0

            pbar.update(len(frames_batch))

        pbar.close()
        cap.release()

        # Calculate PERCLOS
        if total_frames_processed == 0:
            print(f"{Fore.RED}No frames processed{Style.RESET_ALL}")
            return None, None, None

        perclos = (closed_eyes_frames / total_frames_processed) * 100

        # Create DataFrames
        perclos_df = pd.DataFrame({
            'time': [times_batch[-1]],
            'PERCLOS': [perclos]
        })

        yawn_df = pd.DataFrame(yawn_events)

        print(f"{Fore.GREEN}Processing complete:")
        print(f"PERCLOS: {perclos:.2f}%")
        print(f"Yawns detected: {len(yawn_events)}{Style.RESET_ALL}")

        return perclos_df, yawn_df, None

    except Exception as e:
        print(f"{Fore.RED}Error in PERCLOS detection: {str(e)}{Style.RESET_ALL}")
        return None, None, None


def calculate_mar(mouth_points):
    """Calculate MAR using numpy to avoid CUDA limitations"""
    # Vertical distances
    A = np.sqrt((mouth_points[2, 0] - mouth_points[10, 0]) ** 2 +
                (mouth_points[2, 1] - mouth_points[10, 1]) ** 2)
    B = np.sqrt((mouth_points[4, 0] - mouth_points[8, 0]) ** 2 +
                (mouth_points[4, 1] - mouth_points[8, 1]) ** 2)

    # Horizontal distance
    C = np.sqrt((mouth_points[0, 0] - mouth_points[6, 0]) ** 2 +
                (mouth_points[0, 1] - mouth_points[6, 1]) ** 2)

    return (A + B) / (2.0 * C) if C > 0 else 0.0


def get_head_pose_gpu(shape, frame):
    """Calculate head pose estimation using GPU acceleration"""
    try:
        # Move data to GPU
        shape_gpu = cp.asarray(shape)

        # Get image dimensions
        height, width = frame.shape[:2]
        focal_length = width
        center = (width / 2, height / 2)

        # Camera matrix
        camera_matrix = cp.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=cp.float64)

        # Image points
        image_points = cp.array([
            shape[30],  # Nose tip
            shape[8],  # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54]  # Right mouth corner
        ], dtype=cp.float64)

        dist_coeffs = cp.zeros((4, 1))

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            cp.asnumpy(MODEL_POINTS),
            cp.asnumpy(image_points),
            cp.asnumpy(camera_matrix),
            cp.asnumpy(dist_coeffs),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector
        rotation_mat = cv2.Rodrigues(rotation_vec)[0]
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        return euler_angles

    except Exception as e:
        print(f"{Fore.RED}Error in head pose estimation: {str(e)}{Style.RESET_ALL}")
        return None


def detect_perclos_and_yawning(video_path, output_dir, patient, session, start_time_offset=300,
                               duration=180, ear_threshold=0.25, mar_threshold=0.6,
                               yawn_frames=20, consecutive_frames=3, visualize=False):
    """Main function for PERCLOS and yawning detection using GPU acceleration"""
    try:
        # Initialize detector and predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{Fore.RED}Error: Could not open video: {video_path}{Style.RESET_ALL}")
            return None, None, None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            print(f"{Fore.RED}Invalid video properties{Style.RESET_ALL}")
            return None, None, None

        # Calculate frame positions
        start_frame = int(start_time_offset * fps)
        frames_to_process = min(int(duration * fps), total_frames - start_frame)

        # Skip to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize tracking variables
        closed_eyes_frames = 0
        total_frames_processed = 0
        is_yawning = False
        yawn_frames_count = 0
        yawn_start_time = None
        last_pose = None

        # Initialize event lists
        yawn_events = []
        posture_events = []
        perclos_measurements = []

        # Create progress bar
        pbar = tqdm(total=frames_to_process,
                    desc=f"Processing {patient}/{session}",
                    bar_format='{l_bar}{bar:30}{r_bar}')

        # Process frames
        for frame_idx in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break

            total_frames_processed += 1
            current_time = (start_frame + frame_idx) / fps

            # Move frame to GPU
            frame_gpu = cp.asarray(frame)
            gray_gpu = cp.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Detect faces
            faces = detector(cp.asnumpy(gray_gpu))
            if not faces:
                pbar.update(1)
                continue

            # Process each face
            for face in faces:
                shape = predictor(cp.asnumpy(gray_gpu), face)
                shape = face_utils.shape_to_np(shape)

                # Extract eye coordinates
                left_eye = shape[42:48]
                right_eye = shape[36:42]

                # Calculate EAR using GPU
                eye_points = cp.array([left_eye, right_eye])
                ear_output = cp.zeros(2)

                threadsperblock = 256
                blockspergrid = (2 + (threadsperblock - 1)) // threadsperblock

                calculate_ear_gpu[blockspergrid, threadsperblock](
                    cp.array([p[:, 0] for p in eye_points]),
                    cp.array([p[:, 1] for p in eye_points]),
                    ear_output
                )

                ear = float(cp.mean(ear_output))

                # Calculate MAR using GPU
                mouth = shape[48:68]
                mar_output = cp.zeros(1)

                calculate_mar_gpu[blockspergrid, threadsperblock](
                    cp.array([mouth[:, 0]]),
                    cp.array([mouth[:, 1]]),
                    mar_output
                )

                mar = float(mar_output[0])

                # Detect head pose
                euler_angles = get_head_pose_gpu(shape, frame)

                # Process measurements
                if ear < ear_threshold:
                    closed_eyes_frames += 1
                else:
                    if closed_eyes_frames >= consecutive_frames:
                        perclos_measurements.append({
                            'time': current_time,
                            'duration': closed_eyes_frames / fps
                        })
                    closed_eyes_frames = 0

                # Process yawning
                if mar > mar_threshold:
                    yawn_frames_count += 1
                    if yawn_frames_count >= yawn_frames and not is_yawning:
                        yawn_start_time = current_time
                        is_yawning = True
                else:
                    if is_yawning:
                        yawn_events.append({
                            'start_time': yawn_start_time,
                            'end_time': current_time,
                            'duration': current_time - yawn_start_time
                        })
                    is_yawning = False
                    yawn_frames_count = 0

                # Process head pose
                if euler_angles is not None:
                    if last_pose is not None:
                        angle_diff = cp.max(cp.abs(euler_angles - last_pose))
                        if angle_diff > POSTURE_THRESHOLD:
                            posture_events.append({
                                'time': current_time,
                                'angle_difference': float(angle_diff),
                                'pitch': float(euler_angles[0]),
                                'yaw': float(euler_angles[1]),
                                'roll': float(euler_angles[2])
                            })
                    last_pose = euler_angles

            pbar.update(1)

        pbar.close()
        cap.release()

        # Calculate PERCLOS
        if total_frames_processed == 0:
            print(f"{Fore.RED}No frames processed{Style.RESET_ALL}")
            return None, None, None

        perclos = (closed_eyes_frames / total_frames_processed) * 100

        # Create DataFrames
        perclos_df = pd.DataFrame({
            'time': [current_time],
            'PERCLOS': [perclos]
        })

        yawn_df = pd.DataFrame(yawn_events)
        posture_df = pd.DataFrame(posture_events)

        # Print results
        print(f"{Fore.GREEN}Processing complete:")
        print(f"PERCLOS: {perclos:.2f}%")
        print(f"Yawns detected: {len(yawn_events)}")
        print(f"Posture changes: {len(posture_events)}{Style.RESET_ALL}")

        return perclos_df, yawn_df, posture_df

    except Exception as e:
        print(f"{Fore.RED}Error in PERCLOS detection: {str(e)}{Style.RESET_ALL}")
        return None, None, None


def process_blinks_gpu(frame, shape, ear_threshold=0.25):
    """Process blinks using GPU acceleration"""
    try:
        # Extract eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate EAR using GPU
        eye_points = cp.array([left_eye, right_eye])
        ear_output = cp.zeros(2)

        threadsperblock = 256
        blockspergrid = (2 + (threadsperblock - 1)) // threadsperblock

        calculate_ear_gpu[blockspergrid, threadsperblock](
            cp.array([p[:, 0] for p in eye_points]),
            cp.array([p[:, 1] for p in eye_points]),
            ear_output
        )

        ear = float(cp.mean(ear_output))
        return ear < ear_threshold, ear

    except Exception as e:
        print(f"{Fore.RED}Error in blink processing: {str(e)}{Style.RESET_ALL}")
        return False, 0.0


def process_perclos_and_yawning(root_dir, destination_dir, patient, session):
    """Main processing function"""
    try:
        # Create output directories
        for subdir in ['perclos', 'yawning']:
            data_dir = os.path.join(destination_dir, 'data', patient, session, subdir)
            os.makedirs(data_dir, exist_ok=True)

        # Define file paths
        video_file = os.path.join(root_dir, patient, session, f"{session}.avi")
        perclos_file = os.path.join(destination_dir, 'data', patient, session, 'perclos',
                                    f"{session}_perclos.csv")
        yawn_file = os.path.join(destination_dir, 'data', patient, session, 'yawning',
                                 f"{session}_yawning.csv")

        # Check if processing is needed
        if all(os.path.exists(f) for f in [perclos_file, yawn_file]):
            print(f"{Fore.GREEN}Loading existing data...{Style.RESET_ALL}")
            return (pd.read_csv(perclos_file),
                    pd.read_csv(yawn_file),
                    None)

        # Process video
        results = detect_perclos_and_yawning(
            video_path=video_file,
            output_dir=destination_dir,
            patient=patient,
            session=session
        )

        if results is None:
            return None, None, None

        perclos_df, yawn_df, _ = results

        # Save results
        if perclos_df is not None and not perclos_df.empty:
            perclos_df.to_csv(perclos_file, index=False)
        else:
            perclos_df = pd.DataFrame(columns=['time', 'PERCLOS'])

        if yawn_df is not None and not yawn_df.empty:
            yawn_df.to_csv(yawn_file, index=False)
        else:
            yawn_df = pd.DataFrame(columns=['start_time', 'end_time', 'duration'])

        return perclos_df, yawn_df, None

    except Exception as e:
        print(f"{Fore.RED}Error in main processing: {str(e)}{Style.RESET_ALL}")
        return None, None, None

@cuda.jit
def process_eye_state_gpu(landmarks_x, landmarks_y, output_states, ear_threshold):
    """CUDA kernel for parallel eye state processing"""
    idx = cuda.grid(1)
    if idx < landmarks_x.shape[0]:
        # Calculate EAR for left eye
        left_ear = calculate_ear_points(
            landmarks_x[idx, 42:48],
            landmarks_y[idx, 42:48]
        )

        # Calculate EAR for right eye
        right_ear = calculate_ear_points(
            landmarks_x[idx, 36:42],
            landmarks_y[idx, 36:42]
        )

        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0

        # Set eye state (0 for open, 1 for closed)
        output_states[idx] = 1 if avg_ear < ear_threshold else 0

@cuda.jit
def calculate_ear_points(x_coords, y_coords):
    """Helper function for EAR calculation in CUDA kernel"""
    # Vertical distances
    A = cuda.sqrt(
        (x_coords[1] - x_coords[5]) ** 2 +
        (y_coords[1] - y_coords[5]) ** 2
    )
    B = cuda.sqrt(
        (x_coords[2] - x_coords[4]) ** 2 +
        (y_coords[2] - y_coords[4]) ** 2
    )

    # Horizontal distance
    C = cuda.sqrt(
        (x_coords[0] - x_coords[3]) ** 2 +
        (y_coords[0] - y_coords[3]) ** 2
    )

    return (A + B) / (2.0 * C) if C > 0 else 0.0

def process_blinks(root_dir, patient, session, ear_threshold=0.25, consecutive_frames=2):
    """Process blinks using GPU acceleration"""
    try:
        video_file = os.path.join(root_dir, patient, session, f"{session}.avi")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize arrays for batch processing
        batch_size = 32
        landmarks_x = np.zeros((batch_size, 68))
        landmarks_y = np.zeros((batch_size, 68))
        eye_states = np.zeros(batch_size)

        blink_events = []
        blink_counter = 0
        current_blink_start = None
        total_blinks = 0
        batch_count = 0

        # Create progress bar
        pbar = tqdm(total=total_frames,
                    desc=f"Detecting blinks for {patient}/{session}",
                    bar_format='{l_bar}{bar:30}{r_bar}')

        while cap.isOpened():
            batch_landmarks_x = []
            batch_landmarks_y = []
            batch_times = []

            # Process frames in batches
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                if faces:
                    shape = predictor(gray, faces[0])
                    shape = face_utils.shape_to_np(shape)
                    batch_landmarks_x.append(shape[:, 0])
                    batch_landmarks_y.append(shape[:, 1])
                    batch_times.append(current_time)

                pbar.update(1)

            if not batch_landmarks_x:
                break

            # Process batch on GPU
            current_batch_size = len(batch_landmarks_x)
            if current_batch_size > 0:
                # Move data to GPU
                landmarks_x_gpu = cuda.to_device(np.array(batch_landmarks_x))
                landmarks_y_gpu = cuda.to_device(np.array(batch_landmarks_y))
                eye_states_gpu = cuda.to_device(np.zeros(current_batch_size))

                # Configure CUDA kernel
                threadsperblock = 256
                blockspergrid = (current_batch_size + (threadsperblock - 1)) // threadsperblock

                # Process eye states
                process_eye_state_gpu[blockspergrid, threadsperblock](
                    landmarks_x_gpu,
                    landmarks_y_gpu,
                    eye_states_gpu,
                    ear_threshold
                )

                # Get results back from GPU
                eye_states = eye_states_gpu.copy_to_host()

                # Process blink events
                for i, is_closed in enumerate(eye_states):
                    if is_closed:
                        blink_counter += 1
                        if blink_counter >= consecutive_frames and current_blink_start is None:
                            current_blink_start = batch_times[i]
                    else:
                        if current_blink_start is not None:
                            blink_duration = batch_times[i] - current_blink_start
                            blink_events.append({
                                'start_time': current_blink_start,
                                'end_time': batch_times[i],
                                'duration': blink_duration,
                            })
                            total_blinks += 1
                            current_blink_start = None
                        blink_counter = 0

            batch_count += 1

        pbar.close()
        cap.release()

        # Create DataFrame with blink events
        blink_df = pd.DataFrame(blink_events)

        if not blink_df.empty:
            # Calculate blink rate
            video_duration = blink_df['end_time'].max() - blink_df['start_time'].min()
            blink_rate = (total_blinks / video_duration) * 60
            blink_df['blink_rate'] = blink_rate

            # Create output directory and save results
            blink_folder = os.path.join('data', patient, session, 'blinks')
            os.makedirs(blink_folder, exist_ok=True)
            blink_file = os.path.join(blink_folder, f"{session}_blinks.csv")
            blink_df.to_csv(blink_file, index=False)

            print(f"{Fore.GREEN}Blink detection completed:")
            print(f"Total blinks: {total_blinks}")
            print(f"Blink rate: {blink_rate:.2f} blinks/minute")
            print(f"Results saved to: {blink_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No blinks detected{Style.RESET_ALL}")

        return blink_df

    except Exception as e:
        print(f"{Fore.RED}Error in blink detection: {str(e)}{Style.RESET_ALL}")
        return None

def download_shape_predictor():
    """Download facial landmark predictor model"""
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat.bz2"

    if not os.path.exists(filename[:-4]):
        print(f"{Fore.CYAN}Downloading shape predictor...{Style.RESET_ALL}")

        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def update_progress(count, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(url, filename, reporthook=update_progress)

        print(f"{Fore.CYAN}Extracting shape predictor...{Style.RESET_ALL}")
        import bz2
        with bz2.BZ2File(filename) as f:
            data = f.read()
        with open(filename[:-4], 'wb') as f:
            f.write(data)
        os.remove(filename)

    return filename[:-4]

def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on frame"""
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.polylines(frame, [landmarks[36:42]], True, (0, 255, 255), 1)  # Right eye
    cv2.polylines(frame, [landmarks[42:48]], True, (0, 255, 255), 1)  # Left eye
    cv2.polylines(frame, [landmarks[48:60]], True, (0, 0, 255), 1)  # Mouth

def save_event_snapshot(frame, event_type, output_dir, patient, session, timestamp,
                        last_snapshots=None, min_delay=60):
    """Save event snapshots with timestamps"""
    if last_snapshots is None:
        last_snapshots = {}

    if event_type in last_snapshots:
        time_since_last = timestamp - last_snapshots[event_type]
        if time_since_last < min_delay:
            return None, last_snapshots

    event_dir = os.path.join(output_dir, 'snapshots', patient, session, event_type)
    os.makedirs(event_dir, exist_ok=True)

    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%H-%M-%S-%f')[:-3]
    filename = f"{session}_{event_type}_{timestamp_str}.jpg"
    filepath = os.path.join(event_dir, filename)

    frame_copy = frame.copy()
    cv2.putText(frame_copy, f"{event_type.upper()}: {timestamp_str}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(filepath, frame_copy)
    last_snapshots[event_type] = timestamp

    return filepath, last_snapshots