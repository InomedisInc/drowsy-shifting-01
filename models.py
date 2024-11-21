import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from colorama import Fore, Style


def create_model_save_directory(save_path):
    """Create model save directory with proper permissions"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Ensure write permissions
        os.chmod(os.path.dirname(save_path), 0o777)
        return True
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        return False


def save_trained_model(model, save_path, alternate_path=None):
    """Save model with fallback options and proper error handling"""
    if not create_model_save_directory(save_path):
        if alternate_path:
            save_path = alternate_path
            if not create_model_save_directory(alternate_path):
                raise PermissionError(f"Cannot create directory for {alternate_path}")

    try:
        # Try saving in newer .keras format first
        keras_path = save_path.replace('.h5', '.keras')
        model.save(keras_path, save_format='keras')
        return keras_path
    except Exception as e:
        print(f"Error saving in .keras format: {str(e)}")
        try:
            # Fallback to HDF5 format
            model.save(save_path, save_format='h5')
            return save_path
        except Exception as e:
            raise RuntimeError(f"Failed to save model in any format: {str(e)}")

def prepare_sequences(slope_data, crash_data, sequence_length=10, stride=1):
    """
    Prepare sequences for LSTM model with enhanced preprocessing
    """
    X = []
    y = []

    # Ensure slope_data has required columns
    required_columns = ['avg_slope', 'avg_rr']
    if not all(col in slope_data.columns for col in required_columns):
        raise ValueError(f"Slope data missing required columns: {required_columns}")

    # Create time column if not present
    if 'time' not in slope_data.columns:
        slope_data['time'] = slope_data.index.astype(float)

    # Create binary crash labels
    crash_labels = np.zeros(len(slope_data))
    for _, crash in crash_data.iterrows():
        mask = (slope_data['time'] >= crash['start_time']) & (slope_data['time'] <= crash['end_time'])
        crash_labels[mask] = 1

    # Calculate additional features
    slope_data['slope_diff'] = slope_data['avg_slope'].diff()
    slope_data['rr_diff'] = slope_data['avg_rr'].diff()
    slope_data['slope_rolling_mean'] = slope_data['avg_slope'].rolling(window=5).mean()
    slope_data['rr_rolling_mean'] = slope_data['avg_rr'].rolling(window=5).mean()

    # Fill NaN values
    slope_data = slope_data.fillna(method='bfill').fillna(method='ffill')

    # Create sequences
    features = ['avg_slope', 'avg_rr', 'slope_diff', 'rr_diff', 'slope_rolling_mean', 'rr_rolling_mean']
    for i in range(0, len(slope_data) - sequence_length, stride):
        seq = slope_data[features].iloc[i:(i + sequence_length)].values
        X.append(seq)
        y.append(crash_labels[i + sequence_length])

    return np.array(X), np.array(y)


def create_lstm_model(sequence_length, n_features):
    """
    Create enhanced LSTM model architecture with attention mechanism
    """
    # Input layer
    inputs = tf.keras.Input(shape=(sequence_length, n_features))

    # Bidirectional LSTM layers
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(128)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)

    # Apply attention weights
    merged = tf.keras.layers.Multiply()([x, attention_weights])

    # Additional LSTM layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(merged)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Dense layers with batch normalization
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model


def plot_training_history(history, save_path=None):
    """
    Plot training metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics')

    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Plot accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()

    # Plot precision-recall
    axes[1, 1].plot(history.history['precision'], label='Training Precision')
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Precision-Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def cross_validate_model(X, y, n_splits=5):
    """
    Perform k-fold cross-validation with handling for missing classes
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Check if both classes are present in training and validation sets
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print(f"Warning: Fold {fold + 1} has missing classes. Skipping...")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_val_scaled = scaler.transform(X_val_reshaped)

        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)

        # Create and train model
        model = create_lstm_model(X_train.shape[1], X_train.shape[2])

        # Calculate class weights
        n_neg = len(y_train[y_train == 0])
        n_pos = len(y_train[y_train == 1])
        class_weights = {0: 1., 1: n_neg / n_pos if n_pos > 0 else 1.}

        # Train model
        model.fit(
            X_train_scaled, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            class_weight=class_weights,
            verbose=1
        )

        # Evaluate
        y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
        y_pred_proba = model.predict(X_val_scaled)

        # Calculate metrics safely
        try:
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            metrics['accuracy'].append(report['accuracy'])

            # Handle cases where class '1' might be missing
            if '1' in report:
                metrics['precision'].append(report['1']['precision'])
                metrics['recall'].append(report['1']['recall'])
                metrics['f1'].append(report['1']['f1-score'])
            else:
                metrics['precision'].append(0.0)
                metrics['recall'].append(0.0)
                metrics['f1'].append(0.0)

            # Calculate AUC only if both classes are present
            if len(np.unique(y_val)) == 2:
                fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
                metrics['auc'].append(auc(fpr, tpr))
            else:
                metrics['auc'].append(0.0)

        except Exception as e:
            print(f"Warning: Error calculating metrics for fold {fold + 1}: {str(e)}")
            continue

    # Only compute average metrics if we have results
    if not any(metrics.values()):
        raise ValueError("No valid folds found for cross-validation")

    # Print average metrics
    print("\nCross-validation results:")
    for metric, values in metrics.items():
        if values:  # Only print if we have values
            print(f"{metric.capitalize()}: {np.mean(values):.3f} (+/- {np.std(values):.3f})")

    return metrics


def train_crash_detection_model(root_dir, patient_sessions, model_save_path):
    """
    Train LSTM model for crash detection with comprehensive data processing and evaluation

    Parameters:
    root_dir (str): Root directory containing the data
    patient_sessions (list): List of tuples containing (patient, session) pairs
    model_save_path (str): Path to save the trained model

    Returns:
    tuple: (model, history, scaler, cv_metrics)
    """
    all_slope_data = []
    all_crash_data = []

    # Create directories
    plots_dir = os.path.dirname(model_save_path)
    os.makedirs(plots_dir, exist_ok=True)

    # Process each patient session
    for patient, session in patient_sessions:
        save_dir = os.path.join(model_save_path, patient, session)
        os.makedirs(save_dir, exist_ok=True)

        # Load slope data
        slope_file = os.path.join(root_dir, patient, session, 'crash_slope',
                                  f"{session}_crash_slope.csv")
        if not os.path.exists(slope_file):
            print(f"{Fore.YELLOW}Warning: Slope file not found for {patient}/{session}{Style.RESET_ALL}")
            continue

        try:
            slope_data = pd.read_csv(slope_file)
            crash_file = os.path.join(root_dir, patient, session, 'crashes',
                                      f"{session}_crash_episodes.csv")

            if not os.path.exists(crash_file):
                print(f"{Fore.YELLOW}Warning: Crash file not found for {patient}/{session}{Style.RESET_ALL}")
                continue

            crash_data = pd.read_csv(crash_file)
            all_slope_data.append(slope_data)
            all_crash_data.append(crash_data)

        except Exception as e:
            print(f"{Fore.RED}Error processing {patient}/{session}: {str(e)}{Style.RESET_ALL}")
            continue

    if not all_slope_data:
        raise ValueError("No valid data found")

    # Combine data from all sessions
    combined_slope_data = pd.concat(all_slope_data, ignore_index=True)
    combined_crash_data = pd.concat(all_crash_data, ignore_index=True)

    # Prepare sequences for LSTM
    X, y = prepare_sequences(combined_slope_data, combined_crash_data)

    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_metrics = cross_validate_model(X, y)

    # Split data for final training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # Create model
    model = create_lstm_model(X_train.shape[1], X_train.shape[2])

    # Calculate class weights for imbalanced data
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    class_weights = {0: 1., 1: n_neg / n_pos if n_pos > 0 else 1.}

    # Train model with callbacks
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )
        ],
        verbose=1
    )

    # Generate and save plots
    print("\nTraining completed. Plotting results...")
    plot_training_history(history, os.path.join(plots_dir, 'training_history.png'))

    # Evaluate model
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred, os.path.join(plots_dir, 'confusion_matrix.png'))

    # Print evaluation metrics
    print("\nFinal Model Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and scaler with proper error handling
    try:
        alternate_path = os.path.join(os.path.expanduser('~'), 'crash_detection_model')
        saved_path = save_trained_model(model, model_save_path, alternate_path)
        print(f"Model saved successfully to {saved_path}")

        # Save scaler
        scaler_path = os.path.join(os.path.dirname(saved_path), 'scaler.pkl')
        pd.to_pickle(scaler, scaler_path)

    except Exception as e:
        print(f"Error saving model: {str(e)}")
        print("Continuing without saving model...")

    # Save scaler for each patient session
    for patient, session in patient_sessions:
        scaler_dir = os.path.join(plots_dir, patient, session)
        os.makedirs(scaler_dir, exist_ok=True)
        pd.to_pickle(scaler, os.path.join(scaler_dir, 'scaler.pkl'))

    return model, history, scaler, cv_metrics



def predict_crashes(model, scaler, slope_data, sequence_length=10, threshold=0.5):
    """
    Make predictions with confidence scores and additional analysis
    """
    # Continue from previous implementation
    X = np.array(X)

    # Scale features
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Create DataFrame with predictions and confidence
    pred_df = pd.DataFrame({
        'time': slope_data['time'].iloc[sequence_length:].values,
        'crash_probability': predictions.flatten(),
        'predicted_crash': (predictions.flatten() > threshold).astype(int)
    })

    # Add confidence bands
    pred_df['high_confidence'] = pred_df['crash_probability'].apply(
        lambda x: 1 if x > 0.8 else (0 if x < 0.2 else 0.5)
    )

    return pred_df


def evaluate_predictions(true_crashes, predicted_crashes, time_window=5.0):
    """
    Evaluate prediction accuracy with time-based metrics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    detection_delays = []

    # Find matching predictions for each true crash
    for _, true_crash in true_crashes.iterrows():
        crash_start = true_crash['start_time']
        crash_end = true_crash['end_time']

        # Look for predictions within time window
        matched_predictions = predicted_crashes[
            (predicted_crashes['time'] >= crash_start - time_window) &
            (predicted_crashes['time'] <= crash_end + time_window) &
            (predicted_crashes['predicted_crash'] == 1)
            ]

        if len(matched_predictions) > 0:
            true_positives += 1
            # Calculate detection delay
            first_detection = matched_predictions['time'].iloc[0]
            delay = first_detection - crash_start
            detection_delays.append(delay)
        else:
            false_negatives += 1

    # Count unmatched predictions as false positives
    for _, pred in predicted_crashes[predicted_crashes['predicted_crash'] == 1].iterrows():
        pred_time = pred['time']
        is_matched = any(
            (true_crash['start_time'] - time_window <= pred_time <= true_crash['end_time'] + time_window)
            for _, true_crash in true_crashes.iterrows()
        )
        if not is_matched:
            false_positives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    mean_delay = np.mean(detection_delays) if detection_delays else None
    median_delay = np.median(detection_delays) if detection_delays else None

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'mean_detection_delay': mean_delay,
        'median_detection_delay': median_delay,
        'detection_delays': detection_delays
    }


def plot_predictions(slope_data, pred_df, true_crashes=None, save_path=None):
    """
    Visualize predictions with crash probabilities and actual crashes
    """
    plt.figure(figsize=(15, 10))

    # Plot slope data
    plt.subplot(2, 1, 1)
    plt.plot(slope_data['time'], slope_data['slope'], label='Slope', alpha=0.6)
    plt.title('Slope Data and Crash Detection')
    plt.xlabel('Time')
    plt.ylabel('Slope')

    # Highlight actual crashes if provided
    if true_crashes is not None:
        for _, crash in true_crashes.iterrows():
            plt.axvspan(crash['start_time'], crash['end_time'],
                        color='red', alpha=0.2, label='Actual Crash')

    plt.legend()

    # Plot crash probabilities
    plt.subplot(2, 1, 2)
    plt.plot(pred_df['time'], pred_df['crash_probability'],
             label='Crash Probability', color='blue')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')

    # Highlight predictions
    pred_crashes = pred_df[pred_df['predicted_crash'] == 1]
    plt.scatter(pred_crashes['time'], pred_crashes['crash_probability'],
                color='red', label='Predicted Crash', zorder=5)

    plt.title('Crash Probability Over Time')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def analyze_sequence_patterns(model, scaler, test_sequences, true_labels):
    """
    Analyze which sequence patterns lead to crash predictions
    """
    # Get the attention weights from the model
    attention_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('attention').output
    )

    attention_weights = attention_model.predict(test_sequences)

    # Analyze high-attention sequences
    high_attention_seqs = []
    for i, (seq, weights) in enumerate(zip(test_sequences, attention_weights)):
        if true_labels[i] == 1:  # For actual crashes
            max_attention_idx = np.argmax(weights)
            high_attention_seqs.append({
                'sequence': seq[max_attention_idx],
                'attention_weight': weights[max_attention_idx],
                'position': max_attention_idx
            })

    # Calculate average patterns
    if high_attention_seqs:
        avg_pattern = np.mean([s['sequence'] for s in high_attention_seqs], axis=0)
        std_pattern = np.std([s['sequence'] for s in high_attention_seqs], axis=0)

        return {
            'average_pattern': avg_pattern,
            'std_pattern': std_pattern,
            'high_attention_sequences': high_attention_seqs
        }
    return None


def save_model_summary(model, save_path):
    """
    Save model architecture and training parameters
    """
    with open(save_path, 'w') as f:
        # Model architecture
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Training parameters
        f.write('\nTraining Parameters:\n')
        f.write(f"Optimizer: {model.optimizer.__class__.__name__}\n")
        f.write(f"Learning rate: {model.optimizer.learning_rate.numpy()}\n")
        f.write(f"Loss function: {model.loss}\n")

        # Model metrics
        f.write('\nModel Metrics:\n')
        for metric in model.metrics:
            f.write(f"- {metric.name}\n")


def export_predictions(pred_df, save_path):
    """
    Export predictions to CSV with additional metadata
    """
    # Add timestamp
    pred_df['timestamp'] = pd.to_datetime('now')

    # Add confidence categories
    pred_df['confidence_category'] = pred_df['crash_probability'].apply(
        lambda x: 'High' if x > 0.8 else ('Low' if x < 0.2 else 'Medium')
    )

    # Export to CSV
    pred_df.to_csv(save_path, index=False)