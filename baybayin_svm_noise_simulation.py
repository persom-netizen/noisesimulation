"""
CONTROLLED NOISE SIMULATION OF SVM-BASED BAYBAYIN CHARACTER RECOGNITION
Complete Google Colab Implementation

Authors: Rain Melody R. Albao, Duenzel Mae Luna, Las Johansen Caluza
Institution: Leyte Normal University, College of Arts and Sciences
Date: May 2026

This script implements the full research pipeline:
1. Dataset loading and preprocessing
2. HOG feature extraction
3. SVM-RBF training with hyperparameter tuning
4. Controlled noise simulation
5. Performance evaluation
6. Error analysis
7. Visualization for publication
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from skimage.feature import hog
from skimage import io, color, img_as_float
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

class ResearchConfig:
    """Centralized configuration for reproducibility"""
    
    # Dataset
    IMAGE_SIZE = 64  # 64x64 pixels
    CHANNELS = 1  # Grayscale
    NUM_CLASSES = 14  # Baybayin characters (BA, CA, DA, etc.)
    
    # Train/Val/Test split
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # HOG parameters
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    HOG_ORIENTATIONS = 9
    
    # SVM hyperparameter tuning
    C_VALUES = [0.1, 1, 10, 100]
    GAMMA_VALUES = [0.001, 0.01, 0.1, 1]
    CV_FOLDS = 5
    
    # Noise simulation
    NOISE_LEVELS = {
        'Clean': {'gaussian_sigma': 0, 'sp_probability': 0.0},
        'Low': {'gaussian_sigma': 10, 'sp_probability': 0.05},
        'Medium': {'gaussian_sigma': 20, 'sp_probability': 0.10},
        'High': {'gaussian_sigma': 30, 'sp_probability': 0.15},
        'Severe': {'gaussian_sigma': 40, 'sp_probability': 0.20},
    }
    
    # Output directory
    RESULTS_DIR = '/content/results'
    
    RANDOM_STATE = 42

config = ResearchConfig()
np.random.seed(config.RANDOM_STATE)

# ============================================================================
# MODULE 1: DATASET LOADING AND PREPROCESSING
# ============================================================================

class DatasetModule:
    """Handles dataset loading, preprocessing, and standardization"""
    
    def __init__(self, config):
        self.config = config
        self.X_clean = None
        self.y_labels = None
        
    def load_dataset(self, dataset_path):
        """
        Load Baybayin character images from directory structure.
        Expected structure: dataset_path/character_class/image.png
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            X_clean: Array of shape (N_samples, height, width)
            y_labels: Array of shape (N_samples,) with class labels
        """
        print("=" * 70)
        print("MODULE 1: DATASET LOADING AND PREPROCESSING")
        print("=" * 70)
        
        images = []
        labels = []
        class_mapping = {}
        class_id = 0
        
        # Scan dataset directory
        for character_class in sorted(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, character_class)
            if not os.path.isdir(class_path):
                continue
                
            class_mapping[character_class] = class_id
            
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = self._load_and_preprocess_image(img_path)
                        images.append(img)
                        labels.append(class_id)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            class_id += 1
        
        self.X_clean = np.array(images)
        self.y_labels = np.array(labels)
        self.class_mapping = class_mapping
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Total samples: {len(self.X_clean)}")
        print(f"  Image shape: {self.X_clean[0].shape}")
        print(f"  Classes: {list(class_mapping.keys())}")
        
        return self.X_clean, self.y_labels
    
    def _load_and_preprocess_image(self, img_path):
        """
        Load single image and apply preprocessing.
        
        Preprocessing steps:
        1. Load image
        2. Convert to grayscale
        3. Resize to standard dimensions
        4. Normalize intensity [0, 1]
        5. Apply Otsu's thresholding
        """
        # Load image
        img = io.imread(img_path)
        
        # Convert to grayscale if RGB
        if len(img.shape) == 3:
            img = color.rgb2gray(img)
        
        # Resize to standard dimensions
        from skimage.transform import resize
        img = resize(img, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                    anti_aliasing=True)
        
        # Normalize to [0, 1]
        img = img_as_float(img)
        
        # Apply Otsu's thresholding for binarization
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(img)
        img = (img > thresh).astype(float)
        
        return img
    
    def create_train_val_test_split(self, random_state=42):
        """
        Split dataset into training, validation, and test sets.
        
        Split strategy:
        - Training: 70% (for SVM training)
        - Validation: 15% (for hyperparameter tuning via GridSearchCV)
        - Test: 15% (for final robustness evaluation)
        """
        # First split: separate test set (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X_clean, self.y_labels,
            test_size=self.config.TEST_SIZE,
            random_state=random_state,
            stratify=self.y_labels
        )
        
        # Second split: separate training (70%) and validation (15%)
        # From remaining 85%: 70/85 ≈ 0.82, 15/85 ≈ 0.18
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VAL_SIZE / (1 - self.config.TEST_SIZE),
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\n✓ Train/Val/Test split completed")
        print(f"  Training samples: {len(X_train)} ({len(X_train)/len(self.X_clean)*100:.1f}%)")
        print(f"  Validation samples: {len(X_val)} ({len(X_val)/len(self.X_clean)*100:.1f}%)")
        print(f"  Test samples: {len(X_test)} ({len(X_test)/len(self.X_clean)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# MODULE 2: HOG FEATURE EXTRACTION
# ============================================================================

class HOGFeatureExtractor:
    """Extracts HOG features from images"""
    
    def __init__(self, config):
        self.config = config
        
    def extract_hog_features(self, images, verbose=True):
        """
        Extract HOG features from image array.
        
        HOG parameters:
        - Pixels per cell: 8×8
        - Cells per block: 2×2 (for block normalization)
        - Orientations: 9 bins
        
        Output dimensions: 
        - For 64×64 image: (64/8 * 64/8 * 2 * 2 * 9) = 1764 features
        
        Args:
            images: Array of shape (N, 64, 64)
            
        Returns:
            features: Array of shape (N, 1764)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("MODULE 2: HOG FEATURE EXTRACTION")
            print("=" * 70)
        
        features = []
        
        for i, img in enumerate(images):
            # Extract HOG descriptor
            hog_descriptor = hog(
                img,
                orientations=self.config.HOG_ORIENTATIONS,
                pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
                cells_per_block=self.config.HOG_CELLS_PER_BLOCK,
                block_norm='L2-Hys',  # L2-Hys normalization
                visualize=False
            )
            features.append(hog_descriptor)
        
        features = np.array(features)
        
        if verbose:
            print(f"\n✓ HOG feature extraction completed")
            print(f"  Input shape: {images.shape}")
            print(f"  Output shape: {features.shape}")
            print(f"  Features per image: {features.shape[1]}")
            print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
        
        return features


# ============================================================================
# MODULE 3: SVM-RBF TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================

class SVMTrainingEngine:
    """Train and tune SVM-RBF classifier"""
    
    def __init__(self, config):
        self.config = config
        self.svm_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        
    def normalize_features(self, X_train, X_val=None, X_test=None):
        """
        Normalize features to zero mean and unit variance.
        Fit on training data, apply to all sets.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {'X_train': X_train_scaled}
        if X_val is not None:
            results['X_val'] = self.scaler.transform(X_val)
        if X_test is not None:
            results['X_test'] = self.scaler.transform(X_test)
            
        return results
    
    def tune_hyperparameters(self, X_train, y_train, verbose=True):
        """
        Perform grid search for optimal C and gamma parameters.
        
        Grid: C ∈ [0.1, 1, 10, 100], γ ∈ [0.001, 0.01, 0.1, 1]
        Cross-validation: 5-fold stratified
        Scoring metric: F1-score (weighted for multi-class)
        
        Args:
            X_train: Training features (N, 1764)
            y_train: Training labels (N,)
            
        Returns:
            best_params: Dictionary with optimal C and gamma
            grid_search: GridSearchCV object with results
        """
        if verbose:
            print("\n" + "=" * 70)
            print("MODULE 3: HYPERPARAMETER TUNING (GRID SEARCH)")
            print("=" * 70)
        
        param_grid = {
            'C': self.config.C_VALUES,
            'gamma': self.config.GAMMA_VALUES,
        }
        
        svm = SVC(kernel='rbf', random_state=self.config.RANDOM_STATE)
        
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=self.config.CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\nPerforming grid search over {len(param_grid['C']) * len(param_grid['gamma'])} combinations...")
        print(f"Cross-validation folds: {self.config.CV_FOLDS}\n")
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        
        if verbose:
            print(f"\n✓ Grid search completed")
            print(f"  Best C: {self.best_params['C']}")
            print(f"  Best γ: {self.best_params['gamma']:.4f}")
            print(f"  Best CV F1-score: {grid_search.best_score_:.4f}")
        
        return self.best_params, grid_search
    
    def train_final_model(self, X_train, y_train, best_params=None):
        """
        Train final SVM-RBF model with optimal hyperparameters.
        """
        if best_params is None:
            best_params = self.best_params
        
        self.svm_model = SVC(
            kernel='rbf',
            C=best_params['C'],
            gamma=best_params['gamma'],
            random_state=self.config.RANDOM_STATE,
            probability=True  # For ROC-AUC later
        )
        
        print(f"\nTraining final SVM-RBF model with optimal hyperparameters...")
        print(f"  C = {best_params['C']}")
        print(f"  γ = {best_params['gamma']:.4f}\n")
        
        self.svm_model.fit(X_train, y_train)
        
        print(f"✓ Model training completed")
        print(f"  Number of support vectors: {len(self.svm_model.support_vectors_)}")
        
        return self.svm_model


# ============================================================================
# MODULE 4: NOISE SIMULATION ENGINE
# ============================================================================

class NoiseSimulationEngine:
    """Add controlled noise to images"""
    
    def __init__(self, config):
        self.config = config
        
    def add_gaussian_noise(self, image, sigma):
        """
        Add Gaussian noise to image.
        
        Model: x_noisy = x + η, where η ~ N(0, σ²)
        
        Args:
            image: Input image (H, W) with values in [0, 1]
            sigma: Standard deviation of Gaussian noise
            
        Returns:
            noisy_image: Image with Gaussian noise added
        """
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image + noise
        # Clip to valid range [0, 1]
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image
    
    def add_salt_pepper_noise(self, image, probability):
        """
        Add salt-and-pepper noise to image.
        
        Model: 
        - Set p/2 fraction of pixels to white (255)
        - Set p/2 fraction of pixels to black (0)
        - Keep remaining pixels unchanged
        
        Args:
            image: Input image (H, W) with values in [0, 1]
            probability: Probability of noise (0 to 1)
            
        Returns:
            noisy_image: Image with salt-and-pepper noise added
        """
        noisy_image = image.copy()
        random_matrix = np.random.random(image.shape)
        
        # Salt (white, value 1)
        noisy_image[random_matrix < probability/2] = 1
        # Pepper (black, value 0)
        noisy_image[random_matrix >= probability/2] &= (random_matrix < probability/2) | \
                                                        (random_matrix >= probability/2 + probability/2)
        noisy_image[random_matrix >= probability/2 + probability/2] &= (random_matrix >= probability/2)
        noisy_image[random_matrix >= probability/2 + probability/2] = 0 if random_matrix[random_matrix >= probability/2 + probability/2].size > 0 else noisy_image[random_matrix >= probability/2 + probability/2]
        
        # Simpler implementation
        noisy_image = image.copy()
        num_pixels = image.size
        num_noise_pixels = int(num_pixels * probability)
        
        noise_indices = np.random.choice(num_pixels, num_noise_pixels, replace=False)
        noise_image_flat = noisy_image.flatten()
        
        for idx in noise_indices:
            if np.random.random() < 0.5:
                noise_image_flat[idx] = 1  # Salt
            else:
                noise_image_flat[idx] = 0  # Pepper
        
        return noisy_image
    
    def apply_noise_to_dataset(self, images, noise_config):
        """
        Apply combined Gaussian and salt-pepper noise to image dataset.
        
        Args:
            images: Array of shape (N, H, W)
            noise_config: Dict with 'gaussian_sigma' and 'sp_probability'
            
        Returns:
            noisy_images: Array of shape (N, H, W) with noise applied
        """
        noisy_images = []
        
        gaussian_sigma = noise_config['gaussian_sigma']
        sp_probability = noise_config['sp_probability']
        
        for img in images:
            # Apply Gaussian noise first
            if gaussian_sigma > 0:
                img_noisy = self.add_gaussian_noise(img, gaussian_sigma)
            else:
                img_noisy = img.copy()
            
            # Apply salt-pepper noise second
            if sp_probability > 0:
                img_noisy = self.add_salt_pepper_noise(img_noisy, sp_probability)
            
            noisy_images.append(img_noisy)
        
        return np.array(noisy_images)
    
    def simulate_all_noise_levels(self, images):
        """
        Generate images with all noise levels for evaluation.
        
        Returns:
            noise_datasets: Dict with keys = noise levels, values = noisy images
        """
        print("\n" + "=" * 70)
        print("MODULE 4: NOISE SIMULATION ENGINE")
        print("=" * 70)
        
        noise_datasets = {}
        
        for noise_level, noise_config in self.config.NOISE_LEVELS.items():
            print(f"\nApplying {noise_level} noise...")
            print(f"  Gaussian σ: {noise_config['gaussian_sigma']}")
            print(f"  Salt-Pepper p: {noise_config['sp_probability']}")
            
            noisy_images = self.apply_noise_to_dataset(images, noise_config)
            noise_datasets[noise_level] = noisy_images
            
            print(f"  ✓ Completed")
        
        return noise_datasets


# ============================================================================
# MODULE 5: EVALUATION ENGINE
# ============================================================================

class EvaluationEngine:
    """Compute performance metrics across noise levels"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model, X_features, y_true, noise_level_name, scaler=None):
        """
        Evaluate model performance on a dataset.
        
        Metrics:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1-score (weighted)
        
        Args:
            model: Trained SVM model
            X_features: Feature vectors (N, 1764)
            y_true: True labels (N,)
            noise_level_name: Name of noise level
            scaler: Feature scaler
            
        Returns:
            metrics_dict: Dictionary of computed metrics
        """
        if scaler is not None:
            X_features = scaler.transform(X_features)
        
        # Make predictions
        y_pred = model.predict(X_features)
        
        # Compute metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Get confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return metrics, cm, y_pred
    
    def evaluate_robustness(self, model, X_test, y_test, hog_extractor, 
                           noise_engine, scaler):
        """
        Evaluate model robustness across all noise levels.
        
        Args:
            model: Trained SVM model
            X_test: Test images (N, 64, 64)
            y_test: Test labels (N,)
            hog_extractor: HOG feature extractor
            noise_engine: Noise simulation engine
            scaler: Feature scaler
            
        Returns:
            robustness_results: Dict with results for each noise level
        """
        print("\n" + "=" * 70)
        print("MODULE 5: EVALUATION ENGINE - ROBUSTNESS ANALYSIS")
        print("=" * 70)
        
        robustness_results = {}
        
        # Generate noisy test sets
        noise_datasets = noise_engine.simulate_all_noise_levels(X_test)
        
        for noise_level, noisy_images in noise_datasets.items():
            print(f"\nEvaluating on {noise_level} noise level...")
            
            # Extract HOG features from noisy images
            X_features = hog_extractor.extract_hog_features(noisy_images, verbose=False)
            
            # Evaluate
            metrics, cm, y_pred = self.evaluate_model(
                model, X_features, y_test, noise_level, scaler
            )
            
            robustness_results[noise_level] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'predictions': y_pred,
            }
            
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall: {metrics['Recall']:.4f}")
            print(f"  F1-score: {metrics['F1-score']:.4f}")
        
        self.results = robustness_results
        return robustness_results


# ============================================================================
# MODULE 6: ERROR ANALYSIS ENGINE
# ============================================================================

class ErrorAnalysisEngine:
    """Analyze classification errors and confusion patterns"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_confusion_matrix(self, cm, class_names):
        """
        Analyze confusion matrix to identify error patterns.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            
        Returns:
            error_analysis: Dict with error statistics
        """
        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        error_analysis = {}
        
        for i, class_name in enumerate(class_names):
            error_analysis[class_name] = {
                'TP': tp[i],
                'FP': fp[i],
                'FN': fn[i],
                'Precision': tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0,
                'Recall': tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0,
            }
        
        return error_analysis
    
    def identify_confused_pairs(self, cm, class_names, top_k=5):
        """
        Identify most frequently confused character pairs.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            top_k: Number of top misclassifications to return
            
        Returns:
            confused_pairs: List of (class_i, class_j, count) tuples
        """
        confused_pairs = []
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((
                        class_names[i],
                        class_names[j],
                        cm[i, j]
                    ))
        
        confused_pairs = sorted(confused_pairs, key=lambda x: x[2], reverse=True)
        return confused_pairs[:top_k]


# ============================================================================
# MODULE 7: VISUALIZATION DASHBOARD
# ============================================================================

class VisualizationDashboard:
    """Generate publication-ready figures"""
    
    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = output_dir or config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10
        
    def plot_accuracy_vs_noise(self, robustness_results):
        """
        Plot accuracy (and other metrics) vs noise level.
        
        This is the MAIN RESULT FIGURE showing robustness degradation.
        """
        noise_levels = list(robustness_results.keys())
        accuracies = [robustness_results[nl]['metrics']['Accuracy'] for nl in noise_levels]
        precisions = [robustness_results[nl]['metrics']['Precision'] for nl in noise_levels]
        recalls = [robustness_results[nl]['metrics']['Recall'] for nl in noise_levels]
        f1_scores = [robustness_results[nl]['metrics']['F1-score'] for nl in noise_levels]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(noise_levels))
        width = 0.2
        
        ax.bar(x_pos - 1.5*width, accuracies, width, label='Accuracy', color='#2E86AB')
        ax.bar(x_pos - 0.5*width, precisions, width, label='Precision', color='#A23B72')
        ax.bar(x_pos + 0.5*width, recalls, width, label='Recall', color='#F18F01')
        ax.bar(x_pos + 1.5*width, f1_scores, width, label='F1-score', color='#C73E1D')
        
        ax.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Classification Performance vs Noise Level\n(SVM-RBF with HOG Features)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(noise_levels)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_vs_noise.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Saved: accuracy_vs_noise.png")
    
    def plot_confusion_matrices(self, robustness_results, class_names):
        """
        Plot confusion matrices for each noise level.
        """
        fig, axes = plt.subplots(1, len(robustness_results), figsize=(20, 5))
        if len(robustness_results) == 1:
            axes = [axes]
        
        for idx, (noise_level, data) in enumerate(robustness_results.items()):
            cm = data['confusion_matrix']
            
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{noise_level} Noise', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=11)
            axes[idx].set_ylabel('True Label', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Saved: confusion_matrices.png")
    
    def plot_degradation_curve(self, robustness_results):
        """
        Plot performance degradation curve showing accuracy drop with noise.
        """
        noise_levels = list(robustness_results.keys())
        accuracies = [robustness_results[nl]['metrics']['Accuracy'] for nl in noise_levels]
        
        # Calculate degradation rate
        clean_accuracy = accuracies[0]
        degradation_rates = [(clean_accuracy - acc) / clean_accuracy * 100 for acc in accuracies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy curve
        ax1.plot(noise_levels, accuracies, marker='o', linewidth=2.5, 
                markersize=8, color='#2E86AB')
        ax1.fill_between(range(len(noise_levels)), accuracies, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy Under Increasing Noise', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Add value labels on points
        for i, (nl, acc) in enumerate(zip(noise_levels, accuracies)):
            ax1.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=10)
        
        # Plot 2: Degradation rate
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#E63946']
        ax2.bar(noise_levels, degradation_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Degradation Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy Degradation from Clean Baseline', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (nl, rate) in enumerate(zip(noise_levels, degradation_rates)):
            ax2.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'degradation_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Saved: degradation_curve.png")
    
    def create_results_table(self, robustness_results, output_filename='results_table.csv'):
        """
        Create publication-ready results table.
        """
        import pandas as pd
        
        data = []
        for noise_level, result in robustness_results.items():
            metrics = result['metrics']
            data.append({
                'Noise Level': noise_level,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1-score': f"{metrics['F1-score']:.4f}",
            })
        
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(self.output_dir, output_filename)
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Results table saved: {output_filename}")
        print(f"\n{df.to_string(index=False)}")
        
        return df


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_complete_pipeline():
    """
    Execute the complete research pipeline.
    """
    print("\n" + "=" * 70)
    print("CONTROLLED NOISE SIMULATION OF SVM-BASED BAYBAYIN RECOGNITION")
    print("Complete Research Pipeline Execution")
    print("=" * 70)
    
    # NOTE: In actual execution, replace '/path/to/baybayin/dataset' with actual path
    DATASET_PATH = '/path/to/baybayin/dataset'
    
    # ===== MODULE 1: Load Dataset =====
    dataset_loader = DatasetModule(config)
    X_clean, y_labels = dataset_loader.load_dataset(DATASET_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.create_train_val_test_split()
    
    # ===== MODULE 2: Extract HOG Features =====
    hog_extractor = HOGFeatureExtractor(config)
    X_train_hog = hog_extractor.extract_hog_features(X_train, verbose=True)
    X_val_hog = hog_extractor.extract_hog_features(X_val, verbose=False)
    
    # ===== MODULE 3: Train SVM with Hyperparameter Tuning =====
    svm_engine = SVMTrainingEngine(config)
    normalized = svm_engine.normalize_features(X_train_hog, X_val_hog)
    X_train_norm = normalized['X_train']
    X_val_norm = normalized['X_val']
    
    best_params, grid_search_results = svm_engine.tune_hyperparameters(X_train_norm, y_train)
    trained_model = svm_engine.train_final_model(X_train_norm, y_train, best_params)
    
    # ===== MODULE 4: Noise Simulation =====
    noise_engine = NoiseSimulationEngine(config)
    # Noise datasets are generated during evaluation
    
    # ===== MODULE 5: Evaluate Model Robustness =====
    evaluator = EvaluationEngine(config)
    robustness_results = evaluator.evaluate_robustness(
        trained_model, X_test, y_test, hog_extractor, noise_engine, svm_engine.scaler
    )
    
    # ===== MODULE 6: Error Analysis =====
    error_analyzer = ErrorAnalysisEngine(config)
    
    print("\n" + "=" * 70)
    print("MODULE 6: ERROR ANALYSIS")
    print("=" * 70)
    
    for noise_level, result in robustness_results.items():
        print(f"\n{noise_level} Noise:")
        cm = result['confusion_matrix']
        error_analysis = error_analyzer.analyze_confusion_matrix(
            cm, list(dataset_loader.class_mapping.keys())
        )
        
        # Show confused pairs
        confused_pairs = error_analyzer.identify_confused_pairs(
            cm, list(dataset_loader.class_mapping.keys()), top_k=3
        )
        if confused_pairs:
            print("  Top confused character pairs:")
            for class_i, class_j, count in confused_pairs:
                print(f"    {class_i} → {class_j}: {count} times")
    
    # ===== MODULE 7: Visualization =====
    viz = VisualizationDashboard(config)
    
    viz.plot_accuracy_vs_noise(robustness_results)
    viz.plot_confusion_matrices(robustness_results, list(dataset_loader.class_mapping.keys()))
    viz.plot_degradation_curve(robustness_results)
    viz.create_results_table(robustness_results)
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    
    return {
        'model': trained_model,
        'results': robustness_results,
        'scaler': svm_engine.scaler,
        'hog_extractor': hog_extractor,
        'class_mapping': dataset_loader.class_mapping,
    }


if __name__ == "__main__":
    # Execute pipeline
    final_results = run_complete_pipeline()
