import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import tensorflow as tf
from scipy import signal


# ---- models.py ----
from models import prepare_sequences, create_lstm_model


def scale_sequences(X, scaler):
    """
    Scale 3D sequence data using a StandardScaler.

    Args:
        X (numpy.ndarray): Input sequences with shape (samples, timesteps, features)
        scaler (StandardScaler): Fitted sklearn StandardScaler

    Returns:
        numpy.ndarray: Scaled sequences with same shape as input
    """
    # Reshape to 2D for scaling
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Scale the data
    X_scaled = scaler.transform(X_reshaped)

    # Reshape back to 3D
    return X_scaled.reshape(original_shape)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the LSTM model with early stopping and learning rate reduction.

    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        epochs: Maximum number of epochs
        batch_size: Batch size for training

    Returns:
        History object containing training metrics
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]

    # Calculate class weights for imbalanced data
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    class_weights = {0: 1., 1: n_neg / n_pos if n_pos > 0 else 1.}

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return history

def apply_time_shift(rr_data, crash_data, shift_window=(-60, 60), step=5):

    best_metrics = {
        'shift': 0,
        'auc': 0,
        'f1': 0
    }

    shifts = np.arange(shift_window[0], shift_window[1] + step, step)

    for shift in shifts:
        # Create shifted version of RR data
        shifted_rr = rr_data.copy()
        shifted_rr['time'] = shifted_rr['time'] + shift

        # Prepare sequences with shifted data
        X, y = prepare_sequences(shifted_rr, crash_data)

        # Split and normalize data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train_scaled = scale_sequences(X_train, scaler)
        X_val_scaled = scale_sequences(X_val, scaler)

        # Train model with shifted data
        model = create_lstm_model(X_train.shape[1], X_train.shape[2])
        history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val)

        # Evaluate performance
        y_pred = model.predict(X_val_scaled)
        auc = roc_auc_score(y_val, y_pred)
        f1 = f1_score(y_val, (y_pred > 0.5).astype(int))

        if auc > best_metrics['auc']:
            best_metrics = {
                'shift': shift,
                'auc': auc,
                'f1': f1,
                'model': model,
                'scaler': scaler
            }

    return best_metrics


def prepare_sequences_with_filtering(slope_data, crash_data, sequence_length=20):
    """
    Prepares sequences with enhanced filtering and feature extraction.
    """
    X = []
    y = []

    # Calculate additional features
    slope_data['rr_gradient'] = np.gradient(slope_data['RR'])
    slope_data['rr_acceleration'] = np.gradient(slope_data['rr_gradient'])

    # Apply Savitzky-Golay filter for smoothing
    slope_data['rr_smooth'] = signal.savgol_filter(slope_data['RR'],
                                                   window_length=11,
                                                   polyorder=3)

    # Calculate rolling statistics
    slope_data['rr_std'] = slope_data['RR'].rolling(window=10).std()
    slope_data['rr_range'] = slope_data['RR'].rolling(window=10).max() - \
                             slope_data['RR'].rolling(window=10).min()

    # Create binary crash labels with time-based weighting
    crash_labels = np.zeros(len(slope_data))
    time_weights = np.zeros(len(slope_data))

    for _, crash in crash_data.iterrows():
        # Create exponentially decaying weights before crash
        mask = (slope_data['time'] >= crash['start_time'] - 30) & \
               (slope_data['time'] <= crash['end_time'])
        crash_labels[mask] = 1

        # Add time-based weights
        crash_time = crash['start_time']
        times = slope_data.loc[mask, 'time']
        time_diffs = np.abs(times - crash_time)
        weights = np.exp(-time_diffs / 10)  # Exponential decay
        time_weights[mask] = np.maximum(time_weights[mask], weights)

    features = ['RR', 'rr_gradient', 'rr_acceleration', 'rr_smooth',
                'rr_std', 'rr_range']

    # Create sequences with overlap
    stride = 5
    for i in range(0, len(slope_data) - sequence_length, stride):
        seq = slope_data[features].iloc[i:(i + sequence_length)].values
        label_seq = crash_labels[i:(i + sequence_length)]
        weight_seq = time_weights[i:(i + sequence_length)]

        # Only include sequences with sufficient variation
        if np.std(seq[:, 0]) > 0.01:  # Check RR interval variation
            X.append(seq)
            y.append(np.max(label_seq * weight_seq))  # Weight crashes by proximity

    return np.array(X), np.array(y)


def create_enhanced_lstm_model(sequence_length, n_features):
    """
    Creates an enhanced LSTM model with residual connections and attention.
    """
    inputs = tf.keras.Input(shape=(sequence_length, n_features))

    # Multi-scale feature processing
    conv1 = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv2 = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
    conv3 = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)

    # Combine multi-scale features
    conv_merged = tf.keras.layers.Concatenate()([conv1, conv2, conv3])

    # Bidirectional LSTM with residual connections
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(conv_merged)
    lstm1_residual = tf.keras.layers.Add()([conv_merged, lstm1])

    lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(lstm1_residual)
    lstm2_residual = tf.keras.layers.Add()([lstm1_residual, lstm2])

    # Self-attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(
        lstm2_residual, lstm2_residual)
    attention_residual = tf.keras.layers.Add()([lstm2_residual, attention])

    # Global feature extraction
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(attention_residual)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(attention_residual)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])

    # Dense layers with dropout and batch normalization
    dense1 = tf.keras.layers.Dense(64, activation='relu')(pooled)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.4)(dense1)

    dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )

    return model


def train_with_time_shifts(rr_data, crash_data, model_save_path):
    """
    Main function to train model with time shifting.
    """
    # Find optimal time shift
    best_metrics = apply_time_shift(rr_data, crash_data)

    # Apply best time shift to data
    shifted_rr = rr_data.copy()
    shifted_rr['time'] = shifted_rr['time'] + best_metrics['shift']

    # Prepare sequences with enhanced features
    X, y = prepare_sequences_with_filtering(shifted_rr, crash_data)

    # Create and train enhanced model
    model = create_enhanced_lstm_model(X.shape[1], X.shape[2])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scale_sequences(X_train, scaler)
    X_val_scaled = scale_sequences(X_val, scaler)

    # Training with advanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    ]

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight={0: 1, 1: 3}  # Adjust for class imbalance
    )

    return model, history, scaler, best_metrics['shift']