import cv2
import pandas as pd
import os
import numpy as np
from scipy.spatial import distance as dist
import urllib.request
import dlib
from datetime import datetime, timedelta
from imutils import face_utils
from colorama import Fore, Style, init
from tqdm import tqdm


# Initialize colorama
init()

# Add new constants for posture detection
POSTURE_THRESHOLD = 30  # degrees
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])


def get_head_pose(shape, frame):
    """Calculate head pose estimation using facial landmarks."""
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],  # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]  # Right mouth corner
    ], dtype="double")

    # Camera internals
    height, width = frame.shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    # Find rotation and translation vectors
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix and euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angles


def detect_posture_change(frame, shape, last_pose=None):
    """Detect significant changes in head posture."""
    current_pose = get_head_pose(shape, frame)

    if current_pose is None:
        return False, None, 0

    if last_pose is not None:
        # Calculate angle difference
        pitch_diff = abs(current_pose[0] - last_pose[0])
        yaw_diff = abs(current_pose[1] - last_pose[1])
        roll_diff = abs(current_pose[2] - last_pose[2])

        # Check if any angle exceeds threshold
        max_angle_diff = max(pitch_diff, yaw_diff, roll_diff)
        if max_angle_diff > POSTURE_THRESHOLD:
            return True, current_pose, max_angle_diff

    return False, current_pose, 0

def download_shape_predictor():

    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat.bz2"
    if not os.path.exists(filename[:-4]):
        print(f"{Fore.CYAN}Downloading shape predictor file...{Style.RESET_ALL}")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def update_progress(count, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)

            urllib.request.urlretrieve(url, filename, reporthook=update_progress)

        print(f"{Fore.CYAN}Extracting shape predictor file...{Style.RESET_ALL}")
        import bz2
        with bz2.BZ2File(filename) as f:
            data = f.read()
        with open(filename[:-4], 'wb') as f:
            f.write(data)
        os.remove(filename)
    return filename[:-4]


def calculate_ear(eye):

    # Calculate the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Calculate the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth):

    # Calculate the euclidean distances between the vertical mouth landmarks
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # Calculate the euclidean distance between the horizontal mouth landmarks
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar


def detect_blinks(frame, shape, ear_threshold=0.25, consecutive_frames=2):

    # Extract eye coordinates
    left_eye = shape[42:48]
    right_eye = shape[36:42]

    # Calculate eye aspect ratios
    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    ear = (ear_left + ear_right) / 2.0

    return ear < ear_threshold, ear


def draw_landmarks(frame, landmarks):

    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    cv2.polylines(frame, [landmarks[36:42]], True, (0, 255, 255), 1)  # Right eye
    cv2.polylines(frame, [landmarks[42:48]], True, (0, 255, 255), 1)  # Left eye
    cv2.polylines(frame, [landmarks[48:60]], True, (0, 0, 255), 1)  # Mouth


def save_event_snapshot(frame, event_type, output_dir, patient, session, timestamp, last_snapshots=None, min_delay=60):

    if last_snapshots is None:
        last_snapshots = {}

    # Check if enough time has passed since last snapshot of this type
    if event_type in last_snapshots:
        time_since_last = timestamp - last_snapshots[event_type]
        if time_since_last < min_delay:
            return None, last_snapshots

    # Create event-specific directory
    event_dir = os.path.join(output_dir, 'snapshots', patient, session, event_type)
    os.makedirs(event_dir, exist_ok=True)

    # Format timestamp for filename
    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%H-%M-%S-%f')[:-3]
    filename = f"{session}_{event_type}_{timestamp_str}.jpg"
    filepath = os.path.join(event_dir, filename)

    # Add timestamp text to frame
    frame_copy = frame.copy()
    cv2.putText(frame_copy, f"{event_type.upper()}: {timestamp_str}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the frame
    cv2.imwrite(filepath, frame_copy)

    # Update last snapshot time
    last_snapshots[event_type] = timestamp

    return filepath, last_snapshots


def detect_perclos_and_yawning(video_path, output_dir, patient, session, start_time_offset=300,
                               duration=180, ear_threshold=0.25, mar_threshold=0.6, yawn_frames=20,
                               color_threshold=30, consecutive_frames=3, visualize=False,
                               snapshot_min_delay=60):
    # Initialize detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{Fore.RED}Error: Could not open video file: {video_path}{Style.RESET_ALL}")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        print(
            f"{Fore.RED}Error: Invalid video properties. FPS: {fps}, Total frames: {total_frames}{Style.RESET_ALL}")
        return None, None, None

    # Calculate frame positions
    start_frame = int(start_time_offset * fps)
    frames_to_process = min(int(duration * fps), total_frames - start_frame)

    print(f"{Fore.CYAN}Starting detection at {start_time_offset}s for {duration}s duration")
    print(f"Minimum delay between snapshots: {snapshot_min_delay}s{Style.RESET_ALL}")

    # Skip to start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize tracking variables
    closed_eyes_frames = 0
    total_frames_processed = 0
    is_yawning = False
    yawn_frames_count = 0
    yawn_start_time = None
    last_closed_eyes_time = None
    last_pose = None
    last_snapshots = {}

    # Initialize event lists
    yawn_events = []
    posture_events = []
    perclos_measurements = []
    snapshot_counts = {'perclos': 0, 'yawn': 0, 'posture': 0}

    # Create progress bar
    pbar = tqdm(total=frames_to_process,
                desc=f"Processing {patient}/{session}",
                bar_format='{l_bar}{bar:30}{r_bar}')

    # Process frames
    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"{Fore.YELLOW}Warning: Failed to read frame at index {frame_idx}{Style.RESET_ALL}")
            break

        total_frames_processed += 1
        current_time = (start_frame + frame_idx) / fps

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            if visualize:
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Calculate eye aspect ratios
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0

            # Calculate mouth aspect ratio
            mouth = shape[48:68]
            mar = calculate_mar(mouth)

            # Detect posture changes
            posture_changed, current_pose, angle_diff = detect_posture_change(frame, shape, last_pose)
            if posture_changed and current_pose is not None:
                filepath, last_snapshots = save_event_snapshot(
                    frame, 'posture', output_dir, patient, session,
                    current_time, last_snapshots, snapshot_min_delay
                )
                if filepath:
                    snapshot_counts['posture'] += 1

                posture_events.append({
                    'time': current_time,
                    'angle_difference': float(angle_diff),
                    'pitch': float(current_pose[0]),
                    'yaw': float(current_pose[1]),
                    'roll': float(current_pose[2])
                })
            last_pose = current_pose

            # Check for closed eyes
            if ear < ear_threshold:
                closed_eyes_frames += 1
                if last_closed_eyes_time is None:
                    filepath, last_snapshots = save_event_snapshot(
                        frame, 'perclos', output_dir, patient, session,
                        current_time, last_snapshots, snapshot_min_delay
                    )
                    if filepath:
                        snapshot_counts['perclos'] += 1
                    last_closed_eyes_time = current_time
            else:
                if closed_eyes_frames >= consecutive_frames:


                    if last_closed_eyes_time is not None:
                        perclos_measurements.append({
                                'time': last_closed_eyes_time,
                                'duration': float(current_time - last_closed_eyes_time)
                        })
                last_closed_eyes_time = None

            # Check for yawning
            if mar > mar_threshold:
                yawn_frames_count += 1
                if yawn_frames_count >= yawn_frames:
                    if not is_yawning:
                        filepath, last_snapshots = save_event_snapshot(
                            frame, 'yawn', output_dir, patient, session,
                            current_time, last_snapshots, snapshot_min_delay
                        )
                        if filepath:
                            snapshot_counts['yawn'] += 1
                        yawn_start_time = current_time
                    is_yawning = True
            else:
                if is_yawning:
                    yawn_duration = current_time - yawn_start_time
                    yawn_events.append({
                        'start_time': yawn_start_time,
                        'end_time': current_time,
                        'duration': yawn_duration
                    })
                is_yawning = False
                yawn_frames_count = 0

            if visualize:
                draw_landmarks(frame, shape)
                if current_pose is not None:
                    draw_pose_axes(frame, shape, current_pose)

        if visualize:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.update(1)

    pbar.close()
    if visualize:
        cv2.destroyAllWindows()

    if cap.isOpened():
        cap.release()

    # Calculate PERCLOS
    if total_frames_processed == 0:
        print(f"{Fore.RED}Error: No frames were processed{Style.RESET_ALL}")
        return None, None, None

    perclos = (closed_eyes_frames / total_frames_processed) * 100

    # Create DataFrames
    perclos_df = pd.DataFrame({
        'time': [current_time],
        'PERCLOS': [perclos],
        'total_closed_frames': [closed_eyes_frames],
        'total_frames': [total_frames_processed]
    })

    yawn_df = pd.DataFrame(yawn_events)
    posture_df = pd.DataFrame(posture_events)

    print(f"{Fore.GREEN}Results for period {start_time_offset}s to {start_time_offset + duration}s:")
    print(f"PERCLOS: {perclos:.2f}%")
    print(f"Yawns detected: {len(yawn_events)}")
    print(f"Posture changes: {len(posture_events)}")
    print(f"\nSnapshot counts:")
    print(f"PERCLOS snapshots: {snapshot_counts['perclos']}")
    print(f"Yawn snapshots: {snapshot_counts['yawn']}")
    print(f"Posture snapshots: {snapshot_counts['posture']}{Style.RESET_ALL}")

    return perclos_df, yawn_df, posture_df


def draw_pose_axes(frame, shape, euler_angles):

    height, width = frame.shape[:2]
    face_center = np.mean(shape[27:35], axis=0).astype(int)  # Using nose bridge points

    # Scale factor for axis visualization
    scale = 50

    # Convert euler angles to rotation matrix
    pitch, yaw, roll = euler_angles

    # Draw axes
    cv2.line(frame, tuple(face_center),
             tuple(face_center + np.array([0, int(-scale * np.sin(np.deg2rad(pitch)))])),
             (0, 0, 255), 2)  # Pitch (red)
    cv2.line(frame, tuple(face_center),
             tuple(face_center + np.array([int(scale * np.sin(np.deg2rad(yaw))), 0])),
             (0, 255, 0), 2)  # Yaw (green)
    cv2.line(frame, tuple(face_center),
             tuple(face_center + np.array([int(scale * np.sin(np.deg2rad(roll))), 0])),
             (255, 0, 0), 2)  # Roll (blue)


def process_blinks(root_dir, patient, session, ear_threshold=0.25, consecutive_frames=2):

    try:
        video_file = os.path.join(root_dir, patient, session, f"{session}.avi")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        blink_events = []
        blink_counter = 0
        current_blink_start = None
        total_blinks = 0

        # Create progress bar for blink detection
        pbar = tqdm(total=total_frames,
                    desc=f"Detecting blinks for {patient}/{session}",
                    bar_format='{l_bar}{bar:30}{r_bar}')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                is_blinking, ear = detect_blinks(frame, shape, ear_threshold, consecutive_frames)

                if is_blinking:
                    blink_counter += 1
                    if blink_counter >= consecutive_frames and current_blink_start is None:
                        current_blink_start = current_time
                        save_event_snapshot(frame, 'blink', root_dir, patient, session, current_time)
                else:
                    if current_blink_start is not None:
                        blink_duration = current_time - current_blink_start
                        blink_events.append({
                            'start_time': current_blink_start,
                            'end_time': current_time,
                            'duration': blink_duration,
                        })
                        total_blinks += 1
                        current_blink_start = None
                    blink_counter = 0

            # Update progress bar
            pbar.update(1)

        # Close progress bar
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
            print(f"{Fore.YELLOW}No blinks detected in the video{Style.RESET_ALL}")

        return blink_df

    except Exception as e:
        print(f"{Fore.RED}Error in blink detection: {str(e)}{Style.RESET_ALL}")
        return None


def process_perclos_and_yawning(root_dir, destination_dir, patient, session):
    aligned_folder = f"{session}_aligned"
    video_file = os.path.join(root_dir, patient, session, f"{session}.avi")

    # Create directories if they don't exist
    for subdir in ['perclos', 'yawning', 'posture']:
        data_dir = os.path.join(destination_dir, 'data', patient, session, subdir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    # Define file paths
    perclos_file = os.path.join(destination_dir, 'data', patient, session, 'perclos', f"{session}_perclos.csv")
    yawn_file = os.path.join(destination_dir, 'data', patient, session, 'yawning', f"{session}_yawning.csv")
    posture_file = os.path.join(destination_dir, 'data', patient, session, 'posture', f"{session}_posture.csv")

    # Initialize default DataFrames
    default_perclos_df = pd.DataFrame(columns=['time', 'PERCLOS'])
    default_yawn_df = pd.DataFrame(columns=['start_time', 'end_time', 'duration'])
    default_posture_df = pd.DataFrame(columns=['time', 'angle_difference', 'pitch', 'yaw', 'roll'])

    # Initialize output DataFrames with defaults
    perclos_df = default_perclos_df.copy()
    yawn_df = default_yawn_df.copy()
    posture_df = default_posture_df.copy()

    print(f"{Fore.GREEN}Loading existing PERCLOS, yawning, and posture data...{Style.RESET_ALL}")

    # Function to safely read CSV file
    def safe_read_csv(file_path):
        if not os.path.exists(file_path):
            return None
        if os.path.getsize(file_path) == 0:
            return None
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            return df
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error reading {file_path}: {str(e)}{Style.RESET_ALL}")
            return None

    # Load existing data
    temp_perclos = safe_read_csv(perclos_file)
    temp_yawn = safe_read_csv(yawn_file)
    temp_posture = safe_read_csv(posture_file)

    # Process PERCLOS data if valid
    if temp_perclos is not None and 'time' in temp_perclos.columns:
        perclos_df = temp_perclos
        perclos_df['time'] = perclos_df['time'].astype(float)
    else:
        print(f"{Fore.YELLOW}No valid PERCLOS data found{Style.RESET_ALL}")

    # Process yawning data if valid
    if temp_yawn is not None and all(col in temp_yawn.columns for col in ['start_time', 'end_time', 'duration']):
        yawn_df = temp_yawn
        yawn_df['start_time'] = yawn_df['start_time'].astype(float)
        yawn_df['end_time'] = yawn_df['end_time'].astype(float)
    else:
        print(f"{Fore.YELLOW}No valid yawning data found{Style.RESET_ALL}")

    # Process posture data if valid
    if temp_posture is not None and all(
            col in temp_posture.columns for col in ['time', 'angle_difference', 'pitch', 'yaw', 'roll']):
        posture_df = temp_posture
        posture_df['time'] = posture_df['time'].astype(float)
    else:
        print(f"{Fore.YELLOW}No valid posture data found{Style.RESET_ALL}")

    # Check if we need to process new data
    need_processing = (
            perclos_df.equals(default_perclos_df) or
            yawn_df.equals(default_yawn_df) or
            posture_df.equals(default_posture_df)
    )

    if need_processing:
        print(f"{Fore.LIGHTYELLOW_EX}Processing new PERCLOS and yawning detection for {patient}/{session}...")
        print(f"Video file: {video_file}{Style.RESET_ALL}")

        if not os.path.exists(video_file):
            print(f"{Fore.YELLOW}Video file not found: {video_file}{Style.RESET_ALL}")
            return default_perclos_df, default_yawn_df, default_posture_df

        # Initialize face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print(f"{Fore.RED}Failed to load Haarcascade classifier{Style.RESET_ALL}")
            return default_perclos_df, default_yawn_df, default_posture_df

        # Process video
        new_results = detect_perclos_and_yawning(
            video_path=video_file,
            output_dir=destination_dir,
            patient=patient,
            session=session,
            duration=180,
            start_time_offset=180,
            ear_threshold=0.25,
            mar_threshold=0.6,
            yawn_frames=20,
            color_threshold=30,
            consecutive_frames=3,
            visualize=False
        )

        # Update with new data if processing was successful
        if new_results is not None:
            new_perclos_df, new_yawn_df, new_posture_df = new_results

            if new_perclos_df is not None and not new_perclos_df.empty:
                perclos_df = new_perclos_df
                if not perclos_df.empty:
                    perclos_df.to_csv(perclos_file, index=False)

            if new_yawn_df is not None and not new_yawn_df.empty:
                yawn_df = new_yawn_df
                if not yawn_df.empty:
                    yawn_df.to_csv(yawn_file, index=False)

            if new_posture_df is not None and not new_posture_df.empty:
                posture_df = new_posture_df
                if not posture_df.empty:
                    posture_df.to_csv(posture_file, index=False)

    return perclos_df, yawn_df, posture_df

