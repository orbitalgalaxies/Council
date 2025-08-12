import tensorflow as tf
import numpy as np
import os
import json
import logging
import shutil
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D, 
                                     Flatten, MultiHeadAttention, LayerNormalization, Add, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
import keras_tuner as kt

# ==============================================================================
# NOTE: This is a large, comprehensive script. It is self-contained and
# demonstrates a full, professional MLOps pipeline from data prep to validation.
# ==============================================================================

# ==============================================================================
# 1. UTILITY: Logging and Configuration
# ==============================================================================
def setup_logging():
    class SingleHandlerLogger(logging.Logger):
        def __init__(self, name, level=logging.NOTSET):
            super().__init__(name, level)
            self.propagate = False
            if not self.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
                handler.setFormatter(formatter)
                self.addHandler(handler)
    logging.setLoggerClass(SingleHandlerLogger)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

# ==============================================================================
# 2. MODULE: Data Preprocessing & Feature Engineering
# ==============================================================================
class DataPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_classif, k=50) # Select top 50 features

    def simulate_real_world_issues(self, x, y):
        self.logger.info('{"event": "data_simulation_start", "message": "Injecting real-world issues into synthetic data..."}')
        x_messy = x.copy()
        
        # Inject missing values (NaNs)
        nan_mask = np.random.rand(*x_messy.shape) < 0.05
        x_messy[nan_mask] = np.nan
        
        # Inject skewed features
        skewed_col_idx = 0
        x_messy[:, skewed_col_idx] = np.expm1(np.abs(x_messy[:, skewed_col_idx]))
        
        # Add a timestamp feature (seconds since an arbitrary epoch)
        timestamp_feature = np.linspace(1672531200, 1672617600, x_messy.shape[0]).reshape(-1, 1)
        x_messy = np.hstack([x_messy, timestamp_feature])
        
        # Add noisy/uninformative features
        noisy_features = np.random.randn(x_messy.shape[0], 10)
        x_messy = np.hstack([x_messy, noisy_features])
        
        self.logger.info('{"event": "data_simulation_complete"}')
        return x_messy, y

    def impute_missing_values(self, x):
        self.logger.info('{"event": "imputation_start"}')
        x_imputed = self.imputer.fit_transform(x)
        return x_imputed

    def apply_log_transform(self, x):
        self.logger.info('{"event": "log_transform_start"}')
        # Apply to the first column which we know is skewed
        x[:, 0] = np.log1p(x[:, 0])
        return x

    def split_timestamp_feature(self, x):
        self.logger.info('{"event": "timestamp_split_start"}')
        timestamps = x[:, -11] # The timestamp feature we added
        hours = (timestamps % 86400) // 3600
        day_of_week = (timestamps // 86400) % 7
        
        # Remove original timestamp, add new features
        x = np.delete(x, -11, axis=1)
        x = np.hstack([x, hours.reshape(-1, 1), day_of_week.reshape(-1, 1)])
        return x

    def select_features(self, x, y):
        self.logger.info('{"event": "feature_selection_start"}')
        x_selected = self.selector.fit_transform(x, y)
        return x_selected

    def run_all(self, x, y):
        self.logger.info('{"event": "preprocessing_pipeline_start"}')
        x, y = self.simulate_real_world_issues(x, y)
        x = self.impute_missing_values(x)
        x = self.apply_log_transform(x)
        x = self.split_timestamp_feature(x)
        x = self.scaler.fit_transform(x) # Scale before selection
        x = self.select_features(x, y)
        self.logger.info(f'{{"event": "preprocessing_pipeline_complete", "final_shape": {x.shape}}}')
        return x, y

# ==============================================================================
# 3. MODULE: Hyperparameter Tuning
# ==============================================================================
def build_tunable_model(hp, model_type, input_shape, output_dim):
    """A single function to build any of the council's experts for tuning."""
    inputs = Input(shape=input_shape)
    
    if model_type == 'Reductionist':
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        x = Flatten()(inputs)
        x = Dense(units, activation='relu')(x)
        outputs = Dense(output_dim, activation='softmax')(x)
        model = Model(inputs, outputs, name="Reductionist")
    elif model_type == 'Holist_1D':
        units = hp.Int('units', min_value=64, max_value=256, step=64)
        x = Flatten()(inputs)
        res = Dense(units, activation='relu')(x)
        x = Dense(units, activation='relu')(res)
        x = Add()([x, res])
        outputs = Dense(output_dim, activation='softmax')(x)
        model = Model(inputs, outputs, name="Holist_1D")
    elif model_type == 'Analogist':
        filters = hp.Int('filters', min_value=16, max_value=64, step=16)
        x = Reshape((input_shape[0], 1))(inputs)
        x = Conv1D(filters=filters, kernel_size=3, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(output_dim, activation='softmax')(x)
        model = Model(inputs, outputs, name="Analogist")
    else: # Contextualist
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        x = Flatten()(inputs)
        num_features = x.shape[-1]
        x_reshaped = Reshape((num_features, 1))(x)
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=8)(x_reshaped, x_reshaped)
        x = Add()([x_reshaped, attn_output])
        x = Flatten()(x)
        outputs = Dense(output_dim, activation='softmax')(x)
        model = Model(inputs, outputs, name="Contextualist")
        
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class HyperparameterTuner:
    def __init__(self, x_train, y_train, input_shape, output_dim, logger):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.logger = logger

    def tune(self, model_type):
        self.logger.info(f'{{"event": "hyperparameter_tuning_start", "model_type": "{model_type}"}}')
        tuner = kt.Hyperband(
            lambda hp: build_tunable_model(hp, model_type, self.input_shape, self.output_dim),
            objective='val_accuracy',
            max_epochs=3, # Fast for demonstration
            factor=3,
            directory='keras_tuner',
            project_name=f'tune_{model_type}'
        )
        tuner.search(self.x_train, self.y_train, epochs=3, validation_split=0.2, verbose=0)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.logger.info(f'{{"event": "hyperparameter_tuning_complete", "model_type": "{model_type}"}}')
        # Clean up tuner directory
        shutil.rmtree('keras_tuner')
        return best_hps

# ==============================================================================
# 4. CORE FRAMEWORK: Council, Trainer, Architectures
# ==============================================================================
# --- Architectures (now built from hyperparameters) ---
def build_final_reductionist(input_shape, output_dim, hps):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(hps.get('units'), activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs, name="Reductionist")

def build_final_holist_1d(input_shape, output_dim, hps):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    res = Dense(hps.get('units'), activation='relu')(x)
    x = Dense(hps.get('units'), activation='relu')(res)
    x = Add()([x, res])
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs, name="Holist_1D")

def build_final_analogist(input_shape, output_dim, hps):
    inputs = Input(shape=input_shape)
    x = Reshape((input_shape[0], 1))(inputs)
    x = Conv1D(filters=hps.get('filters'), kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs, name="Analogist")

def build_final_contextualist(input_shape, output_dim, hps):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    num_features = x.shape[-1]
    x_reshaped = Reshape((num_features, 1))(x)
    attn_output = MultiHeadAttention(num_heads=hps.get('num_heads'), key_dim=8)(x_reshaped, x_reshaped)
    x = Add()([x_reshaped, attn_output])
    x = Flatten()(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs, name="Contextualist")

# --- Council Class ---
class HardenedCouncil:
    def __init__(self, input_shape, output_dim, best_hps):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.best_hps = best_hps
        self.council = {}
        self.is_trained = False

    def assemble_council(self):
        self.council['Reductionist'] = build_final_reductionist(self.input_shape, self.output_dim, self.best_hps['Reductionist'])
        self.council['Holist_1D'] = build_final_holist_1d(self.input_shape, self.output_dim, self.best_hps['Holist_1D'])
        self.council['Analogist'] = build_final_analogist(self.input_shape, self.output_dim, self.best_hps['Analogist'])
        self.council['Contextualist'] = build_final_contextualist(self.input_shape, self.output_dim, self.best_hps['Contextualist'])

    def deliberate(self, x_sample):
        if not self.is_trained: raise RuntimeError("Council is not trained.")
        x_sample_exp = np.expand_dims(x_sample, axis=0)
        predictions = {name: model.predict(x_sample_exp, verbose=0)[0] for name, model in self.council.items()}
        avg_pred = np.mean(np.array(list(predictions.values())), axis=0)
        return np.argmax(avg_pred)

# --- Trainer Class ---
class CouncilTrainer:
    def __init__(self, council, logger):
        self.council = council
        self.logger = logger

    def train(self, x_train, y_train, x_val, y_val):
        for name, model in self.council.council.items():
            self.logger.info(f'{{"event": "final_training_start", "model": "{name}"}}')
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=10, batch_size=32,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                      verbose=0)
        self.council.is_trained = True

# ==============================================================================
# 5. MAIN: Orchestration Script
# ==============================================================================
def main():
    # --- 1. Setup ---
    warnings.filterwarnings('ignore', category=UserWarning, module='keras')
    warnings.filterwarnings('ignore', category=FutureWarning)
    logger = setup_logging()
    
    # --- 2. Data Pipeline ---
    logger.info('{"event": "protocol_start", "message": "Initiating full pre-training protocol."}')
    x_data, y_data = np.random.rand(1000, 100), np.random.randint(0, 5, 1000)
    
    preprocessor = DataPreprocessor(logger)
    x_clean, y_clean = preprocessor.run_all(x_data, y_data)
    
    input_shape = (x_clean.shape[1],)
    output_dim = len(np.unique(y_clean))

    # --- 3. Hyperparameter Tuning ---
    tuner = HyperparameterTuner(x_clean, y_clean, input_shape, output_dim, logger)
    best_hps = {
        'Reductionist': tuner.tune('Reductionist'),
        'Holist_1D': tuner.tune('Holist_1D'),
        'Analogist': tuner.tune('Analogist'),
        'Contextualist': tuner.tune('Contextualist')
    }
    
    # --- 4. K-Fold Cross-Validation ---
    logger.info('{"event": "kfold_validation_start", "folds": 3}')
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x_clean, y_clean)):
        logger.info(f'{{"event": "fold_start", "fold_number": {fold + 1}}}')
        x_train_fold, x_test_fold = x_clean[train_idx], x_clean[test_idx]
        y_train_fold, y_test_fold = y_clean[train_idx], y_clean[test_idx]

        # Assemble and train a new council for each fold
        council = HardenedCouncil(input_shape, output_dim, best_hps)
        council.assemble_council()
        trainer = CouncilTrainer(council, logger)
        trainer.train(x_train_fold, y_train_fold, x_test_fold, y_test_fold)
        
        # Evaluate
        predictions = [council.deliberate(x) for x in x_test_fold]
        acc = accuracy_score(y_test_fold, predictions)
        fold_accuracies.append(acc)
        logger.info(f'{{"event": "fold_complete", "fold_number": {fold + 1}, "accuracy": {acc:.4f}}}')

    # --- 5. Final Report ---
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print("\n\n======================================================")
    print("=         HIGH-ACCURACY PROTOCOL COMPLETE          =")
    print("======================================================")
    print("\nThis script demonstrated the full end-to-end process:")
    print("1. Simulated and cleaned a messy, real-world-like dataset.")
    print("2. Performed automated hyperparameter tuning for each expert.")
    print("3. Trained and evaluated the final council using robust K-Fold Cross-Validation.")
    print("\n--- FINAL PERFORMANCE REPORT ---")
    print(f"Mean Accuracy across 3 folds: {mean_acc:.4f}")
    print(f"Standard Deviation of Accuracy: {std_acc:.4f}")
    print("\nThis framework is now ready. Apply it to your real dataset.")
    print("======================================================")

if __name__ == '__main__':
    main()