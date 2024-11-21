import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cupy as cp
from colorama import Fore, Style

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{Fore.GREEN}GPU memory growth configured{Style.RESET_ALL}")
    except RuntimeError as e:
        print(f"{Fore.RED}Error configuring GPU: {e}{Style.RESET_ALL}")


def prepare_sequences_gpu(slope_data, crash_data, sequence_length=10, stride=1):
    """Prepare sequences using GPU acceleration"""
    try:
        # Move data to GPU memory
        slope_array = cp.asarray(slope_data['slope'].values)
        time_array = cp.asarray(slope_data['time'].values)

        # Create crash labels on GPU
        crash_labels = cp.zeros(len(slope_data))
        for _, crash in crash_data.iterrows():
            mask = (time_array >= crash['start_time']) & (time_array <= crash['end_time'])
            crash_labels[mask] = 1

        # Calculate features on GPU
        slope_diff = cp.diff(slope_array)
        slope_diff = cp.pad(slope_diff, (1, 0), 'edge')

        rr_array = cp.asarray(slope_data['RR'].values)
        rr_diff = cp.diff(rr_array)
        rr_diff = cp.pad(rr_diff, (1, 0), 'edge')

        # Calculate rolling means
        def rolling_mean_gpu(arr, window=5):
            kernel = cp.ones(window) / window
            return cp.convolve(arr, kernel, mode='same')

        slope_rolling_mean = rolling_mean_gpu(slope_array)
        rr_rolling_mean = rolling_mean_gpu(rr_array)

        # Create sequences
        X = []
        y = []

        for i in range(0, len(slope_data) - sequence_length, stride):
            features = cp.stack([
                slope_array[i:i + sequence_length],
                rr_array[i:i + sequence_length],
                slope_diff[i:i + sequence_length],
                rr_diff[i:i + sequence_length],
                slope_rolling_mean[i:i + sequence_length],
                rr_rolling_mean[i:i + sequence_length]
            ], axis=1)

            X.append(cp.asnumpy(features))
            y.append(float(crash_labels[i + sequence_length]))

        return np.array(X), np.array(y)

    except Exception as e:
        print(f"{Fore.RED}Error in GPU sequence preparation: {str(e)}{Style.RESET_ALL}")
        return None, None


def create_gpu_model(sequence_length, n_features):
    """Create GPU-optimized LSTM model"""
    try:
        with tf.device('/GPU:0'):
            # Input layer
            inputs = tf.keras.Input(shape=(sequence_length, n_features))

            # Bidirectional LSTM layers with CuDNNLSTM
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(inputs)
            x = tf.keras.layers.Dropout(0.3)(x)

            # Attention mechanism
            attention = tf.keras.layers.Dense(1, activation='tanh')(x)
            attention = tf.keras.layers.Flatten()(attention)
            attention_weights = tf.keras.layers.Activation('softmax')(attention)
            attention_weights = tf.keras.layers.RepeatVector(128)(attention_weights)
            attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)

            # Apply attention
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

            # Use mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall')]
            )

            return model

    except Exception as e:
        print(f"{Fore.RED}Error creating GPU model: {str(e)}{Style.RESET_ALL}")
        return None


def predict_crashes(model, scaler, slope_data, sequence_length=10, threshold=0.5):
    """Make predictions with confidence scores using GPU acceleration"""
    try:
        # Prepare sequences
        X, _ = prepare_sequences_gpu(slope_data, pd.DataFrame(), sequence_length)
        if X is None:
            raise ValueError("Failed to prepare sequences")

        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)

        # Make predictions using GPU
        with tf.device('/GPU:0'):
            predictions = model.predict(X_scaled, batch_size=64)

        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            'time': slope_data['time'].iloc[sequence_length:].values,
            'crash_probability': predictions.flatten(),
            'predicted_crash': (predictions.flatten() > threshold).astype(int)
        })

        # Add confidence bands
        pred_df['confidence'] = pred_df['crash_probability'].apply(
            lambda x: 'High' if x > 0.8 else ('Low' if x < 0.2 else 'Medium')
        )

        # Add additional metrics
        pred_df['prediction_confidence'] = np.abs(pred_df['crash_probability'] - 0.5) * 2

        return pred_df

    except Exception as e:
        print(f"{Fore.RED}Error in crash prediction: {str(e)}{Style.RESET_ALL}")
        return None


def evaluate_predictions(true_crashes, predicted_crashes, time_window=5.0):
    """Evaluate predictions with comprehensive metrics"""
    try:
        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'detection_delays': [],
            'early_warnings': [],
            'confidence_scores': []
        }

        # Process true crashes
        for _, true_crash in true_crashes.iterrows():
            crash_start = true_crash['start_time']
            crash_end = true_crash['end_time']

            # Find predictions within time window
            matched_predictions = predicted_crashes[
                (predicted_crashes['time'] >= crash_start - time_window) &
                (predicted_crashes['time'] <= crash_end + time_window) &
                (predicted_crashes['predicted_crash'] == 1)
                ]

            if not matched_predictions.empty:
                metrics['true_positives'] += 1

                # Calculate detection timing
                first_detection = matched_predictions['time'].iloc[0]
                delay = first_detection - crash_start
                metrics['detection_delays'].append(delay)

                if delay < 0:
                    metrics['early_warnings'].append(abs(delay))

                # Record confidence scores
                metrics['confidence_scores'].extend(
                    matched_predictions['prediction_confidence'].tolist()
                )
            else:
                metrics['false_negatives'] += 1

        # Count false positives
        for _, pred in predicted_crashes[predicted_crashes['predicted_crash'] == 1].iterrows():
            pred_time = pred['time']
            is_matched = any(
                (true_crash['start_time'] - time_window <= pred_time <=
                 true_crash['end_time'] + time_window)
                for _, true_crash in true_crashes.iterrows()
            )
            if not is_matched:
                metrics['false_positives'] += 1

        # Calculate performance metrics
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']

        metrics.update({
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'mean_detection_delay': np.mean(metrics['detection_delays'])
            if metrics['detection_delays'] else None,
            'median_detection_delay': np.median(metrics['detection_delays'])
            if metrics['detection_delays'] else None,
            'mean_early_warning': np.mean(metrics['early_warnings'])
            if metrics['early_warnings'] else None,
            'mean_confidence': np.mean(metrics['confidence_scores'])
            if metrics['confidence_scores'] else None
        })

        metrics['f1_score'] = (
            2 * (metrics['precision'] * metrics['recall']) /
            (metrics['precision'] + metrics['recall'])
            if (metrics['precision'] + metrics['recall']) > 0 else 0
        )

        return metrics

    except Exception as e:
        print(f"{Fore.RED}Error in prediction evaluation: {str(e)}{Style.RESET_ALL}")
        return None


def plot_predictions(slope_data, pred_df, true_crashes=None, save_path=None):
    """Create comprehensive visualization of predictions"""
    try:
        plt.style.use('seaborn')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        # Plot 1: Slope Data and Crashes
        ax1.plot(slope_data['time'], slope_data['slope'],
                 label='Slope', color='blue', alpha=0.6)
        ax1.set_title('Slope Data and Crash Detection', fontsize=12)
        ax1.set_ylabel('Slope')

        if true_crashes is not None:
            for _, crash in true_crashes.iterrows():
                ax1.axvspan(crash['start_time'], crash['end_time'],
                            color='red', alpha=0.2, label='Actual Crash')

        # Plot 2: Crash Probability
        ax2.plot(pred_df['time'], pred_df['crash_probability'],
                 label='Crash Probability', color='purple')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
        ax2.set_title('Crash Probability', fontsize=12)
        ax2.set_ylabel('Probability')

        # Highlight predictions
        pred_crashes = pred_df[pred_df['predicted_crash'] == 1]
        ax2.scatter(pred_crashes['time'], pred_crashes['crash_probability'],
                    color='red', label='Predicted Crash', zorder=5)

        # Plot 3: Confidence Levels
        confidence_colors = {
            'High': 'green',
            'Medium': 'yellow',
            'Low': 'red'
        }

        for confidence in ['High', 'Medium', 'Low']:
            mask = pred_df['confidence'] == confidence
            ax3.scatter(pred_df[mask]['time'],
                        pred_df[mask]['prediction_confidence'],
                        c=confidence_colors[confidence],
                        label=f'{confidence} Confidence',
                        alpha=0.6)

        ax3.set_title('Prediction Confidence', fontsize=12)
        ax3.set_ylabel('Confidence Score')
        ax3.set_xlabel('Time')

        # Add legends
        for ax in [ax1, ax2, ax3]:
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{Fore.GREEN}Plot saved to {save_path}{Style.RESET_ALL}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"{Fore.RED}Error in plotting predictions: {str(e)}{Style.RESET_ALL}")


def save_model_summary(model, save_path):
    """Save detailed model architecture and training parameters"""
    try:
        with open(save_path, 'w') as f:
            # Model architecture
            f.write("Model Architecture:\n")
            f.write("=" * 50 + "\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))

            # Training parameters
            f.write("\nTraining Parameters:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Optimizer: {model.optimizer.__class__.__name__}\n")
            f.write(f"Learning rate: {model.optimizer.learning_rate.numpy()}\n")
            f.write(f"Loss function: {model.loss}\n")

            # Model metrics
            f.write("\nModel Metrics:\n")
            f.write("=" * 50 + "\n")
            for metric in model.metrics:
                f.write(f"- {metric.name}\n")

            print(f"{Fore.GREEN}Model summary saved to {save_path}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error saving model summary: {str(e)}{Style.RESET_ALL}")


def export_predictions(pred_df, save_path):
    """Export predictions with detailed metadata"""
    try:
        # Add timestamp
        pred_df['timestamp'] = pd.to_datetime('now')

        # Add confidence categories
        pred_df['confidence_category'] = pred_df['crash_probability'].apply(
            lambda x: 'High' if x > 0.8 else ('Low' if x < 0.2 else 'Medium')
        )

        # Add prediction strength
        pred_df['prediction_strength'] = np.abs(pred_df['crash_probability'] - 0.5) * 2

        # Export to CSV
        pred_df.to_csv(save_path, index=False)
        print(f"{Fore.GREEN}Predictions exported to {save_path}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error exporting predictions: {str(e)}{Style.RESET_ALL}")


def train_crash_detection_model(root_dir, patient_sessions, model_save_path):
    """Train the crash detection model using GPU acceleration"""
    try:
        all_slope_data = []
        all_crash_data = []

        print(f"{Fore.CYAN}Loading and preprocessing data...{Style.RESET_ALL}")
        for patient, session in patient_sessions:
            slope_file = os.path.join(root_dir, 'data', patient, session, 'crash_slope',
                                      f"{session}_crash_slope.csv")
            if not os.path.exists(slope_file):
                print(f"{Fore.YELLOW}Warning: Slope file not found for {patient}/{session}{Style.RESET_ALL}")
                continue

            crash_file = os.path.join(root_dir, 'data', patient, session, 'crashes',
                                      f"{session}_crash_episodes.csv")
            if not os.path.exists(crash_file):
                print(f"{Fore.YELLOW}Warning: Crash file not found for {patient}/{session}{Style.RESET_ALL}")
                continue

            try:
                slope_data = pd.read_csv(slope_file)
                crash_data = pd.read_csv(crash_file)

                all_slope_data.append(slope_data)
                all_crash_data.append(crash_data)
            except Exception as e:
                print(f"{Fore.RED}Error loading data for {patient}/{session}: {str(e)}{Style.RESET_ALL}")
                continue

        if not all_slope_data:
            raise ValueError("No valid data found")

        # Combine data
        print(f"{Fore.CYAN}Combining data...{Style.RESET_ALL}")
        combined_slope_data = pd.concat(all_slope_data, ignore_index=True)
        combined_crash_data = pd.concat(all_crash_data, ignore_index=True)

        # Prepare sequences using GPU
        print(f"{Fore.CYAN}Preparing sequences...{Style.RESET_ALL}")
        X, y = prepare_sequences_gpu(combined_slope_data, combined_crash_data)
        if X is None or y is None:
            raise ValueError("Failed to prepare sequences")

        # Perform cross-validation
        print(f"{Fore.CYAN}Performing cross-validation...{Style.RESET_ALL}")
        cv_metrics = cross_validate_model_gpu(X, y)
        if cv_metrics is None:
            raise ValueError("Cross-validation failed")

        # Final model training
        print(f"{Fore.CYAN}Training final model...{Style.RESET_ALL}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        # Create and train final model
        model = create_gpu_model(X_train.shape[1], X_train.shape[2])
        if model is None:
            raise ValueError("Failed to create model")

        # Handle class imbalance
        class_weights = {
            0: 1.,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }

        # Train model with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_save_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        print(f"{Fore.CYAN}Starting model training...{Style.RESET_ALL}")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate final model
        print(f"{Fore.CYAN}Evaluating model...{Style.RESET_ALL}")
        y_pred = predict_crashes(model, scaler, pd.DataFrame({
            'time': range(len(X_test_scaled)),
            'slope': X_test_scaled[:, 0, 0]
        }))

        if y_pred is not None:
            y_pred_binary = (y_pred['crash_probability'].values > 0.5).astype(int)

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_binary))

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred['crash_probability'].values)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")

            roc_path = os.path.join(os.path.dirname(model_save_path), 'roc_curve.png')
            plt.savefig(roc_path)
            plt.close()

        # Save training history
        plot_training_history(history,
                              save_path=os.path.join(os.path.dirname(model_save_path),
                                                     'training_history.png'))

        # Save model and scaler
        model_dir = os.path.dirname(model_save_path)
        os.makedirs(model_dir, exist_ok=True)

        model.save(model_save_path)
        pd.to_pickle(scaler, os.path.join(model_dir, 'scaler.pkl'))

        print(f"{Fore.GREEN}Model training completed successfully!")
        print(f"Model saved to: {model_save_path}")
        print(f"Scaler saved to: {os.path.join(model_dir, 'scaler.pkl')}{Style.RESET_ALL}")

        return model, history, scaler, cv_metrics

    except Exception as e:
        print(f"{Fore.RED}Error in model training: {str(e)}{Style.RESET_ALL}")
        return None, None, None, None


def cross_validate_model_gpu(X, y, n_splits=5):
    """Perform k-fold cross-validation using GPU acceleration"""
    try:
        print(f"{Fore.CYAN}Starting {n_splits}-fold cross-validation...{Style.RESET_ALL}")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n{Fore.CYAN}Training fold {fold + 1}/{n_splits}{Style.RESET_ALL}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_val_scaled = scaler.transform(X_val_reshaped)

            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)

            # Create and train model
            model = create_gpu_model(X_train.shape[1], X_train.shape[2])
            if model is None:
                print(f"{Fore.YELLOW}Warning: Skipping fold {fold + 1} due to model creation failure{Style.RESET_ALL}")
                continue

            # Handle class imbalance
            class_weights = {
                0: 1.,
                1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            }

            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=30,
                batch_size=32,
                validation_data=(X_val_scaled, y_val),
                class_weight=class_weights,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ]
            )

            # Evaluate
            y_pred = model.predict(X_val_scaled)
            y_pred_binary = (y_pred > 0.5).astype(int)

            report = classification_report(y_val, y_pred_binary, output_dict=True)
            fpr, tpr, _ = roc_curve(y_val, y_pred)

            metrics['accuracy'].append(report['accuracy'])
            metrics['precision'].append(report['1']['precision'])
            metrics['recall'].append(report['1']['recall'])
            metrics['f1'].append(report['1']['f1-score'])
            metrics['auc'].append(auc(fpr, tpr))

            print(f"\n{Fore.GREEN}Fold {fold + 1} Results:")
            print(f"Accuracy: {metrics['accuracy'][-1]:.3f}")
            print(f"Precision: {metrics['precision'][-1]:.3f}")
            print(f"Recall: {metrics['recall'][-1]:.3f}")
            print(f"F1-Score: {metrics['f1'][-1]:.3f}")
            print(f"AUC: {metrics['auc'][-1]:.3f}{Style.RESET_ALL}")

        # Print average metrics
        print(f"\n{Fore.GREEN}Cross-validation Results:")
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.capitalize()}: {mean_val:.3f} (+/- {std_val:.3f}){Style.RESET_ALL}")

        return metrics

    except Exception as e:
        print(f"{Fore.RED}Error in cross-validation: {str(e)}{Style.RESET_ALL}")
        return None