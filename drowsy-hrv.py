import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def process_hrv_data(hrv_signal, sampling_rate=256):
    """
    Process HRV signal and extract features including derivatives
    """
    # Detrending to remove baseline wander
    detrended = signal.detrend(hrv_signal)

    # Calculate first derivative (rate of change)
    first_derivative = np.gradient(detrended, 1 / sampling_rate)

    # Calculate second derivative (rate of rate of change)
    second_derivative = np.gradient(first_derivative, 1 / sampling_rate)

    # Time domain features
    features = {}

    # Basic HRV features
    features['mean_hrv'] = np.mean(detrended)
    features['std_hrv'] = np.std(detrended)
    features['rmssd'] = np.sqrt(np.mean(np.diff(detrended) ** 2))

    # First derivative features
    features['mean_first_deriv'] = np.mean(np.abs(first_derivative))
    features['std_first_deriv'] = np.std(first_derivative)
    features['max_first_deriv'] = np.max(np.abs(first_derivative))

    # Second derivative features
    features['mean_second_deriv'] = np.mean(np.abs(second_derivative))
    features['std_second_deriv'] = np.std(second_derivative)
    features['max_second_deriv'] = np.max(np.abs(second_derivative))

    # Frequency domain features using Welch's method
    frequencies, psd = signal.welch(detrended, fs=sampling_rate)

    # VLF (0-0.04 Hz), LF (0.04-0.15 Hz), HF (0.15-0.4 Hz)
    vlf_mask = (frequencies >= 0) & (frequencies < 0.04)
    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)

    features['vlf_power'] = np.trapz(psd[vlf_mask])
    features['lf_power'] = np.trapz(psd[lf_mask])
    features['hf_power'] = np.trapz(psd[hf_mask])
    features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']

    return features


def train_drowsiness_detector(X, y):
    """
    Train a Random Forest classifier for drowsiness detection
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    return clf, scaler


def detect_drowsiness(hrv_signal, clf, scaler, sampling_rate=256):
    """
    Detect drowsiness from a new HRV signal
    """
    # Extract features
    features = process_hrv_data(hrv_signal, sampling_rate)

    # Convert to array and scale
    X = np.array([[v for v in features.values()]])
    X_scaled = scaler.transform(X)

    # Predict
    prediction = clf.predict(X_scaled)
    probability = clf.predict_proba(X_scaled)

    return prediction[0], probability[0]