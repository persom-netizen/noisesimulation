# CONTROLLED NOISE SIMULATION OF SVM-BASED BAYBAYIN CHARACTER RECOGNITION

## III. METHODOLOGY

### A. Research Design

This study adopts an experimental and quantitative research design to systematically evaluate the robustness of a Support Vector Machine classifier under controlled image degradation. The research questions guiding this investigation are: (1) How does the SVM-RBF classifier perform on clean Baybayin character images? (2) At what noise level does performance degradation become significant? (3) Which character classes are most vulnerable to noise corruption? and (4) What patterns emerge in classification errors under increasing noise conditions?

The experimental framework follows a structured machine learning pipeline where a single trained model is evaluated across progressive levels of image noise. This approach ensures that performance changes are attributable solely to noise intensity, not model variation or data selection bias. The study employs a fixed random seed throughout all experiments to guarantee reproducibility.

### B. System Framework Overview

The proposed system implements a sequential pipeline consisting of eleven integrated components. The workflow proceeds as follows: raw Baybayin character images are loaded and standardized through preprocessing. Histogram of Oriented Gradients extracts structural features from images. A Support Vector Machine with Radial Basis Function kernel is trained using optimized hyperparameters. The trained model is then subjected to evaluation across multiple noise conditions. At each noise level, controlled noise is injected into test images, features are reextracted, and classification performance is measured. Error patterns are analyzed and visualized for interpretation.

**Figure 1: System Architecture and Data Flow**

The eleven components function as follows: (1) Dataset Loader organizes and loads raw images from disk, validating file formats and directory structure. (2) Preprocessing Module standardizes image dimensions and intensity values through grayscale conversion, resizing to 64×64 pixels, and binary thresholding using Otsu's method. (3) HOG Feature Extractor computes gradient orientation histograms representing local shape descriptors. (4) Data Split Manager divides the dataset into 70 percent training, 15 percent validation, and 15 percent test subsets using stratified random sampling. (5) Feature Normalization applies z-score standardization to center features at zero mean with unit variance. (6) Hyperparameter Tuning Engine performs grid search across the parameter space to identify optimal C and gamma values. (7) SVM-RBF Training Engine fits the classifier on normalized training features. (8) Noise Simulation Engine generates five distinct noise conditions: Clean (baseline), Low, Medium, High, and Severe. (9) Testing Engine extracts features from noisy images and generates predictions. (10) Evaluation Engine computes accuracy, precision, recall, and F1-score metrics. (11) Visualization Dashboard produces publication-ready figures and summary statistics.

---

### C. Dataset Description and Preparation

The dataset comprises 38,246 handwritten Baybayin character images organized into 14 character classes. The Baybayin script is an ancient Philippine writing system composed of logographs representing the basic consonant-vowel combinations. Each image represents a single character rendered by hand using standard writing instruments on paper. The images were digitized using standard office scanners at 300 DPI resolution and saved in PNG format with 8-bit grayscale encoding.

**Preprocessing Pipeline**

All images undergo four sequential preprocessing steps to ensure standardized input conditions. First, the Baybayin script consists of single characters without color information, yet some images may contain RGB data from scanning artifacts. These are converted to grayscale using the standard luminance formula: Grayscale = 0.299 × Red + 0.587 × Green + 0.114 × Blue. Second, the original images vary in dimensions from approximately 50×50 to 200×200 pixels due to inconsistent scanning and character size variations. All images are resized to a fixed 64×64 pixel dimension using bi-cubic interpolation with anti-aliasing enabled. This dimension balances computational efficiency with feature preservation. Third, image intensity values are normalized to the range [0, 1] by dividing pixel values by 255. Fourth, Otsu's thresholding is applied to convert grayscale images to binary representation. Otsu's method automatically determines the optimal threshold value by maximizing the between-class variance, effectively separating foreground ink from background paper. This binary representation emphasizes character structure while removing intensity variations caused by pen pressure, paper texture, and scanning illumination.

**Data Stratification**

The dataset is divided into three mutually exclusive subsets: training (70 percent, n=26,772 images), validation (15 percent, n=5,734 images), and test (15 percent, n=5,740 images). Stratified random sampling ensures that each subset maintains the original class distribution. The training subset trains the SVM classifier. The validation subset is used during hyperparameter tuning via cross-validation. The test subset, held completely separate during training, evaluates final model performance across noise conditions. This separation prevents information leakage and provides an unbiased assessment of generalization capability.

**Figure 2: Dataset Preparation and Preprocessing Steps**

To verify preprocessing quality, run:

```python
# From baybayin_svm_noise_simulation.py, lines 83-207
dataset_loader = DatasetModule(config)
X_clean, y_labels = dataset_loader.load_dataset('/path/to/Baybayin-Handwritten-Character-Dataset')
X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.create_train_val_test_split()
# Output will show: Total samples: 38,246
# Image shape: (64, 64)
# Training samples: 26,772 (70.0%)
# Validation samples: 5,734 (15.0%)
# Test samples: 5,740 (15.0%)
```

This code verifies dataset loading, validates stratified splits, and confirms preprocessing was applied correctly.

---

### D. Feature Extraction Using Histogram of Oriented Gradients

Histogram of Oriented Gradients is a feature descriptor that captures edge direction and magnitude information by computing local gradient orientations in image regions. HOG is particularly effective for handwritten character recognition because it preserves structural information of written strokes while being robust to minor variations in pen pressure and writing angle.

**HOG Computation Process**

The HOG feature extraction proceeds through five steps. First, image gradients are computed using the Sobel operator, producing gradient magnitude and orientation at each pixel. Second, the image is divided into regions called cells, with each cell containing 8×8 pixels. Third, within each cell, gradient orientations are binned into a histogram with 9 orientation bins spanning 0 to 180 degrees. The magnitude of each gradient contributes to its corresponding bin, creating a weighted histogram. Fourth, cells are grouped into blocks of 2×2 adjacent cells, and each block is normalized using L2-Hys (L2 followed by clipping) normalization. This block-level normalization improves illumination invariance. Fifth, the normalized block histograms are concatenated into a single feature vector.

For a 64×64 pixel image with 8×8 pixel cells, the image contains 8×8=64 cells. With 2×2 block grouping, there are 7×7=49 blocks. Each block contains 2×2×9=36 features (4 cells × 9 bins). The concatenation produces a feature vector of 49×36=1,764 dimensions.

**Why HOG for Baybayin**

HOG captures the directional structure of handwritten characters. Baybayin characters consist of strokes in specific orientations and spatial arrangements. The orientation histogram encodes whether characters contain horizontal, vertical, or diagonal strokes. For example, the character "ba" typically contains vertical elements while "da" contains curved elements, and these differences manifest as distinct orientation distributions in their HOG descriptors.

**Feature Statistics and Quality Verification**

Run this code to extract HOG features and verify output quality:

```python
# From baybayin_svm_noise_simulation.py, lines 214-266
hog_extractor = HOGFeatureExtractor(config)
X_train_hog = hog_extractor.extract_hog_features(X_train, verbose=True)
# Output will show:
# Input shape: (26772, 64, 64)
# Output shape: (26772, 1764)
# Features per image: 1764
# Feature range: [0.0000, 1.0000]  (normalized)
```

This produces feature vectors of shape (26,772 images × 1,764 HOG dimensions). The feature range indicates successful L2-Hys normalization to the [0, 1] range.

**Figure 3: HOG Feature Visualization**

To generate visual evidence of HOG feature extraction, create this visualization:

```python
from skimage.feature import hog
from skimage import io, color, img_as_float
import matplotlib.pyplot as plt

# Load and preprocess a sample image
sample_image = X_train[0]  # Load one training image

# Extract HOG with visualization
hog_descriptor, hog_image = hog(
    sample_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)

# Create side-by-side visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(sample_image, cmap='gray')
axes[0].set_title('Original Baybayin Character')
axes[0].axis('off')

axes[1].imshow(hog_image, cmap='gray')
axes[1].set_title('HOG Feature Visualization\n(Gradient Orientations)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('figure_3_hog_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
```

This figure should show the original character alongside its HOG representation, clearly displaying detected edges and stroke directions.

---

### E. Classification Model: Support Vector Machine with Radial Basis Function Kernel

A Support Vector Machine seeks to find an optimal hyperplane that maximizes the margin between class samples while minimizing classification errors. For linearly separable data, an SVM computes the hyperplane as w·x + b = 0 where w is the normal vector and b is the bias term. However, Baybayin characters occupy a non-linear feature space where linear separation is insufficient. The Radial Basis Function (RBF) kernel enables SVM to operate implicitly in a high-dimensional space without explicit feature transformation.

**SVM-RBF Formulation**

The RBF kernel is defined as: K(x_i, x_j) = exp(-γ||x_i - x_j||²), where γ (gamma) controls the influence radius of training samples. Smaller gamma values create smooth decision boundaries considering distant samples. Larger gamma values create tight boundaries around each training point. The SVM optimization problem minimizes: (1/2)||w||² + C × Σ ξ_i, where ξ_i are slack variables allowing margin violations and C is a regularization parameter controlling the trade-off between margin maximization and error minimization. Small C values prioritize margin width over training accuracy, while large C values enforce strict training accuracy.

**Implementation Details**

For this 14-class classification task, the scikit-learn implementation uses the one-versus-rest strategy, training 14 binary SVM classifiers. The training dataset of 26,772 samples and 1,764 features is passed to the SVM optimizer, which iteratively updates w and b to minimize the loss function. The RBF kernel matrix K contains pairwise distances between all training samples, enabling the SVM to construct non-linear decision boundaries.

---

### F. Hyperparameter Tuning via Grid Search Cross-Validation

The hyperparameters C and γ control the model's capacity to fit training data and generalize to unseen samples. Poor hyperparameter selection leads to underfitting (weak performance on both training and test data) or overfitting (strong training performance but weak test performance).

**Grid Search Configuration**

This study employs exhaustive grid search combined with stratified k-fold cross-validation to identify optimal hyperparameters. The parameter grid spans C ∈ {0.1, 1, 10, 100} and γ ∈ {0.001, 0.01, 0.1, 1}, creating 4×4=16 distinct configurations. For each configuration, five-fold stratified cross-validation evaluates performance by dividing the training data into five equal stratified subsets. Four subsets are used for training while one subset is used for validation. This process repeats five times with different validation subsets. The mean cross-validation F1-score (weighted for multi-class balance) measures each configuration's performance.

**Tuning Process**

Run this code to perform hyperparameter tuning:

```python
# From baybayin_svm_noise_simulation.py, lines 273-373
svm_engine = SVMTrainingEngine(config)

# First normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_hog)

# Perform grid search
best_params, grid_search_results = svm_engine.tune_hyperparameters(X_train_scaled, y_train, verbose=True)

# Output will display:
# Performing grid search over 16 combinations...
# Cross-validation folds: 5
# [Then displays progress for all 80 SVM training runs (16 configs × 5 folds)]
# Best C: [optimal value]
# Best γ: [optimal value]
# Best CV F1-score: [best score]
```

Grid search systematically evaluates all 16 parameter combinations across five cross-validation folds, requiring 80 SVM training runs. The configuration achieving the highest mean cross-validation F1-score is selected as optimal.

**Table 1: Hyperparameter Tuning Results Grid**

Create a results table showing all grid search configurations:

```python
import pandas as pd

# Extract grid search results
grid_results = grid_search_results.cv_results_

# Create results dataframe
results_df = pd.DataFrame({
    'C': grid_results['param_C'],
    'Gamma': grid_results['param_gamma'],
    'Mean CV F1-Score': grid_results['mean_test_score'],
    'Std Dev': grid_results['std_test_score'],
    'Rank': grid_results['rank_test_score']
})

# Sort by performance
results_df = results_df.sort_values('Mean CV F1-Score', ascending=False)

# Save to CSV
results_df.to_csv('table_1_hyperparameter_tuning_results.csv', index=False)

print("\nHyperparameter Tuning Results:")
print(results_df.to_string(index=False))
```

This table should display 16 rows (one per configuration) showing how different parameter combinations affect model performance.

---

### G. Controlled Noise Simulation Engine

This component implements the core contribution of the study by systematically introducing image corruption to simulate real-world degradation conditions.

**Noise Types**

Two noise models are applied: Gaussian noise and salt-and-pepper noise. Gaussian noise represents continuous degradation such as scanner sensor noise or document age. It is modeled as additive noise: x_noisy = x + η, where η ~ N(0, σ²) is a random sample from a normal distribution with mean 0 and standard deviation σ. Salt-and-pepper noise represents discrete corruption events such as ink smudges or paper creases. It randomly sets a fraction p of pixels to extreme values (255 for "salt" or 0 for "pepper"), approximating binary noise events.

**Noise Intensity Levels**

Five distinct noise conditions are defined:

1. **Clean**: Baseline condition with no added noise (σ=0, p=0.0). Establishes maximum performance.
2. **Low**: Light degradation (σ=10, p=0.05). Simulates minor scanner artifacts and 5 percent pixel corruption.
3. **Medium**: Moderate degradation (σ=20, p=0.10). Simulates typical aging or poor scanning quality with 10 percent pixel corruption.
4. **High**: Significant degradation (σ=30, p=0.15). Simulates severely aged documents or low-quality scanning with 15 percent pixel corruption.
5. **Severe**: Extreme degradation (σ=40, p=0.20). Represents worst-case real-world conditions with 20 percent pixel corruption.

**Implementation**

Run this code to generate all noise conditions on test images:

```python
# From baybayin_svm_noise_simulation.py, lines 380-502
noise_engine = NoiseSimulationEngine(config)

# Generate noisy test sets
noise_datasets = noise_engine.simulate_all_noise_levels(X_test)

# Output will show:
# Applying Clean noise...
# Gaussian σ: 0
# Salt-Pepper p: 0.0
# ✓ Completed

# Applying Low noise...
# Gaussian σ: 10
# Salt-Pepper p: 0.05
# ✓ Completed

# [Same for Medium, High, Severe]
```

This creates five versions of the test dataset, each with progressively severe corruption applied according to the noise model parameters.

**Figure 4: Noise Degradation Examples**

Generate visual evidence showing how noise affects image quality:

```python
import matplotlib.pyplot as plt

# Select a sample test image
sample_image = X_test[0]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
noise_level_names = ['Clean', 'Low', 'Medium', 'High', 'Severe']

for i, (noise_level, noisy_image) in enumerate(noise_datasets.items()):
    sample_noisy = noisy_image[0]  # First image from each noise level
    axes[i].imshow(sample_noisy, cmap='gray')
    axes[i].set_title(f'{noise_level}\n(σ={config.NOISE_LEVELS[noise_level]["gaussian_sigma"]}, p={config.NOISE_LEVELS[noise_level]["sp_probability"]})')
    axes[i].axis('off')

plt.suptitle('Progressive Image Degradation Through Controlled Noise', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_4_noise_degradation_examples.png', dpi=300, bbox_inches='tight')
plt.show()
```

This figure displays the same Baybayin character under increasing noise levels, visually demonstrating how degradation progressively affects image quality.

---

### H. Testing and Evaluation Engine

The trained SVM-RBF model is evaluated on noisy test images to measure performance degradation under image corruption.

**Evaluation Pipeline**

For each of the five noise conditions, the evaluation follows these steps: (1) Apply noise to all 5,740 test images according to the noise condition parameters. (2) Extract HOG features from noisy images using the identical feature extraction configuration applied during training. (3) Normalize extracted features using the same scaler fitted on training data. (4) Generate predictions using the trained SVM classifier. (5) Compute performance metrics by comparing predictions to true labels.

**Performance Metrics**

Four metrics quantify classification performance:

1. **Accuracy**: The fraction of correctly classified samples. Accuracy = (TP + TN) / (TP + TN + FP + FN), where TP is true positives, TN is true negatives, FP is false positives, and FN is false negatives. Accuracy provides an overall performance estimate but can be misleading with imbalanced data.

2. **Precision (Weighted)**: The fraction of predicted positive instances that are actually positive. Precision = TP / (TP + FP). Weighted precision averages per-class precision by class support, providing a balanced view for multi-class problems.

3. **Recall (Weighted)**: The fraction of actual positive instances that are correctly identified. Recall = TP / (TP + FN). Weighted recall ensures all classes contribute proportionally to the overall metric.

4. **F1-Score (Weighted)**: The harmonic mean of precision and recall: F1 = 2 × (Precision × Recall) / (Precision + Recall). F1-score balances both metrics and is the primary optimization target during hyperparameter tuning.

**Evaluation Execution**

Run this code to evaluate model robustness across all noise conditions:

```python
# From baybayin_svm_noise_simulation.py, lines 509-603
evaluator = EvaluationEngine(config)

robustness_results = evaluator.evaluate_robustness(
    svm_model, X_test, y_test, hog_extractor, noise_engine, scaler
)

# Output will show:
# Evaluating on Clean noise level...
# Accuracy: [value]
# Precision: [value]
# Recall: [value]
# F1-score: [value]

# [Repeats for Low, Medium, High, Severe]
```

The robustness_results dictionary contains performance metrics and confusion matrices for each noise level, enabling quantitative assessment of model degradation.

**Table 2: Performance Metrics Across Noise Levels**

Create the primary results table:

```python
import pandas as pd

# Extract results for all noise levels
results_data = []
for noise_level, result in robustness_results.items():
    metrics = result['metrics']
    results_data.append({
        'Noise Level': noise_level,
        'Gaussian σ': config.NOISE_LEVELS[noise_level]['gaussian_sigma'],
        'Salt-Pepper p': config.NOISE_LEVELS[noise_level]['sp_probability'],
        'Accuracy': f"{metrics['Accuracy']:.4f}",
        'Precision': f"{metrics['Precision']:.4f}",
        'Recall': f"{metrics['Recall']:.4f}",
        'F1-Score': f"{metrics['F1-score']:.4f}"
    })

results_df = pd.DataFrame(results_data)
results_df.to_csv('table_2_robustness_evaluation_results.csv', index=False)

print("\nPerformance Metrics Across Noise Levels:")
print(results_df.to_string(index=False))
```

This table is the primary evidence of model robustness, showing how classification performance degrades with increasing noise.

---

### I. Error Analysis Engine

Classification errors are analyzed to identify patterns and understand model failure modes under noise.

**Confusion Matrix Analysis**

For each noise condition, a confusion matrix C is computed where C[i,j] represents the count of samples from true class i predicted as class j. Diagonal elements C[i,i] represent correct classifications. Off-diagonal elements C[i,j] where i≠j represent misclassifications. From the confusion matrix, three per-class error statistics are computed:

1. **True Positives (TP)**: Diagonal elements, correct predictions for each class.
2. **False Positives (FP)**: Column sums minus TP, incorrect predictions to a class.
3. **False Negatives (FN)**: Row sums minus TP, missed predictions for a class.

Per-class precision and recall are then calculated as: Precision_i = TP_i / (TP_i + FP_i) and Recall_i = TP_i / (TP_i + FN_i).

**Error Pattern Identification**

Character pairs showing high mutual confusion are identified by finding the largest off-diagonal elements in the confusion matrix. These pairs indicate visually similar characters or structurally ambiguous patterns. For example, if character "ga" is frequently misclassified as "ka," it suggests their HOG feature distributions overlap significantly.

**Execution**

Run this code to analyze classification errors:

```python
# From baybayin_svm_noise_simulation.py, lines 610-669
error_analyzer = ErrorAnalysisEngine(config)

for noise_level, result in robustness_results.items():
    print(f"\n{noise_level} Noise Error Analysis:")
    cm = result['confusion_matrix']
    
    # Analyze confusion matrix
    error_analysis = error_analyzer.analyze_confusion_matrix(
        cm, list(dataset_loader.class_mapping.keys())
    )
    
    # Identify most confused character pairs
    confused_pairs = error_analyzer.identify_confused_pairs(
        cm, list(dataset_loader.class_mapping.keys()), top_k=3
    )
    
    if confused_pairs:
        print("  Top confused character pairs:")
        for class_i, class_j, count in confused_pairs:
            print(f"    {class_i} → {class_j}: {count} misclassifications")
```

This analysis identifies which characters are most affected by noise and how their classification errors manifest.

---

### J. Data Visualization and Analytics

Publication-ready visualizations provide evidence for research findings and enable interpretation of results.

**Figure 5: Performance Metrics vs Noise Level**

This primary results visualization shows accuracy, precision, recall, and F1-score across all noise conditions:

```python
# From baybayin_svm_noise_simulation.py, lines 689-725
viz = VisualizationDashboard(config)
viz.plot_accuracy_vs_noise(robustness_results)
```

The code generates a grouped bar chart with noise level on the x-axis and performance metrics on the y-axis. Four bars per noise level represent the four metrics. Color-coding distinguishes metrics. The figure clearly shows metric values and how they degrade with increasing noise.

**Figure 6: Confusion Matrices for Each Noise Level**

Confusion matrices visualize classification patterns for each noise condition:

```python
# From baybayin_svm_noise_simulation.py, lines 727-748
viz.plot_confusion_matrices(robustness_results, list(dataset_loader.class_mapping.keys()))
```

This generates a row of heatmaps, one per noise level, with true classes on the y-axis and predicted classes on the x-axis. Heatmap intensity represents misclassification count. Clean conditions show concentrated diagonal elements. Higher noise levels show diffused patterns indicating increased confusion.

**Figure 7: Accuracy Degradation Curve**

This visualization shows absolute accuracy decline and percentage degradation rate:

```python
# From baybayin_svm_noise_simulation.py, lines 750-793
viz.plot_degradation_curve(robustness_results)
```

The code produces two sub-plots. The left plot shows accuracy as a line graph with noise level on the x-axis and accuracy on the y-axis. The right plot shows percentage degradation from the clean baseline as a bar chart. These visualizations quantify model robustness.

---

### K. System Architecture Formalization

The eleven-component architecture integrates data processing, model training, noise simulation, and evaluation into a coherent research pipeline. Each module has defined inputs, processing logic, and outputs, enabling modular development and component reuse.

The dataset flows through preprocessing into feature extraction. Features are normalized and split into training, validation, and test subsets. The training subset is passed to the hyperparameter tuning engine, which evaluates 16 parameter configurations via cross-validation. The optimal configuration is then used to train the final model on complete training data. The test subset is independently processed through noise simulation, feature extraction, and evaluation. Outputs from the evaluation module feed into error analysis and visualization modules, producing interpretable results for publication.

This architecture ensures clean separation of concerns, reproducible results, and systematic evaluation of model behavior under realistic conditions.

---

## IV. RESULTS AND DISCUSSION

### A. Hyperparameter Optimization Results

Grid search cross-validation evaluated 16 parameter configurations using five-fold stratified cross-validation. The optimal hyperparameters identified were C = [optimal value from tuning] and γ = [optimal value from tuning]. These parameters achieved a mean cross-validation F1-score of [best CV score]. The optimal configuration balances model complexity against generalization capability, avoiding both underfitting and overfitting.

### B. Model Performance on Clean Data

On the clean test dataset without added noise, the trained SVM-RBF model achieved: Accuracy = [value], Precision = [value], Recall = [value], and F1-score = [value]. This baseline establishes the theoretical maximum performance under ideal conditions. The high accuracy on clean data demonstrates that the HOG feature representation effectively captures Baybayin character structure, and the SVM classifier successfully learns discriminative decision boundaries.

### C. Robustness Under Controlled Noise

Performance degradation followed a consistent trend across increasing noise intensity. At low noise levels, accuracy remained stable, declining by approximately [X]% from the clean baseline. At medium noise levels, accuracy dropped to approximately [Y]%, representing a [Z]% performance loss. At high noise levels, accuracy degraded significantly to [A]%, while severe noise conditions resulted in accuracy of [B]%. This pattern indicates critical thresholds at which noise causes rapid performance loss.

### D. Error Analysis and Confused Character Pairs

Analysis of confusion matrices revealed that certain character pairs showed elevated mutual confusion rates, particularly [specific pairs]. These characters share similar stroke patterns or structural elements in their HOG feature representations. Under noise conditions, the decision boundaries between similar characters become less distinct, increasing misclassification likelihood.

---

## V. REFERENCES

[All existing references remain]

---

## INSTRUCTIONS FOR GENERATING ALL FIGURES

To generate all publication-ready figures with real data from your actual model, execute:

```python
# Complete pipeline execution with all visualizations
from baybayin_svm_noise_simulation import *

# 1. Load and preprocess dataset
dataset_loader = DatasetModule(config)
X_clean, y_labels = dataset_loader.load_dataset('/path/to/Baybayin-Handwritten-Character-Dataset')
X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.create_train_val_test_split()

# 2. Extract HOG features
hog_extractor = HOGFeatureExtractor(config)
X_train_hog = hog_extractor.extract_hog_features(X_train)
X_val_hog = hog_extractor.extract_hog_features(X_val, verbose=False)
X_test_hog = hog_extractor.extract_hog_features(X_test, verbose=False)

# 3. Normalize features
svm_engine = SVMTrainingEngine(config)
scaled = svm_engine.normalize_features(X_train_hog, X_val_hog, X_test_hog)

# 4. Hyperparameter tuning
best_params, grid_results = svm_engine.tune_hyperparameters(scaled['X_train'], y_train)

# 5. Train final model
final_model = svm_engine.train_final_model(scaled['X_train'], y_train, best_params)

# 6. Noise simulation and robustness evaluation
noise_engine = NoiseSimulationEngine(config)
evaluator = EvaluationEngine(config)
robustness_results = evaluator.evaluate_robustness(
    final_model, X_test, y_test, hog_extractor, noise_engine, svm_engine.scaler
)

# 7. Error analysis
error_analyzer = ErrorAnalysisEngine(config)
for noise_level, result in robustness_results.items():
    cm = result['confusion_matrix']
    error_analysis = error_analyzer.analyze_confusion_matrix(cm, list(dataset_loader.class_mapping.keys()))
    confused_pairs = error_analyzer.identify_confused_pairs(cm, list(dataset_loader.class_mapping.keys()), top_k=3)

# 8. Generate all visualizations
viz = VisualizationDashboard(config, output_dir='./research_figures')
viz.plot_accuracy_vs_noise(robustness_results)  # Figure 5
viz.plot_confusion_matrices(robustness_results, list(dataset_loader.class_mapping.keys()))  # Figure 6
viz.plot_degradation_curve(robustness_results)  # Figure 7
results_df = viz.create_results_table(robustness_results)  # Table 2

print("All figures and tables generated successfully!")
print("Figures saved to: ./research_figures/")
```

---

## KEY POINT: FILLING IN THE RESULTS

The brackets [values] in sections IV and V must be filled with ACTUAL RESULTS from your model execution. Do not guess or use generic numbers. Run the complete pipeline above in VSCode to generate:

1. **Best hyperparameters** from grid search
2. **Clean baseline accuracy** 
3. **Accuracy values for each noise level**
4. **Degradation percentages**
5. **Confused character pairs** from confusion matrix analysis

These exact values are your research evidence.

