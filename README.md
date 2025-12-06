# Chronic Kidney Disease Classification using k-NN
## CSCI 31022 - Machine Learning and Pattern Recognition - Assignment 1

### Project Overview
This project implements a comprehensive k-Nearest Neighbors (k-NN) classifier for predicting chronic kidney disease (CKD) using clinical data. The implementation includes complete data preprocessing, exploratory data analysis, feature engineering, dimensionality reduction, hyperparameter tuning, and model evaluation.

### Dataset Information
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 397 (248 CKD, 149 Not CKD)
- **Features**: 24 attributes (11 numeric, 14 nominal)
- **Target**: Binary classification (ckd/notckd)

### Key Attributes
- **Numeric**: age, blood pressure, blood glucose random, blood urea, serum creatinine, sodium, potassium, hemoglobin, packed cell volume, white blood cell count, red blood cell count
- **Categorical**: specific gravity, albumin, sugar, red blood cells, pus cell, pus cell clumps, bacteria, hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, anemia

### Installation

#### Prerequisites
- Python 3.7 or higher
- pip package manager

#### Install Required Packages
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Usage

#### Run Complete Analysis
```bash
python3 knn_ckd_classification.py
```

This will execute the entire pipeline and generate:
1. Multiple visualization plots (PNG files)
2. Comprehensive text report (classification_report.txt)
3. Console output with detailed analysis

### Implementation Steps

The code performs the following comprehensive analysis:

#### 1. **Data Loading**
   - Parses ARFF file format
   - Handles missing values marked as '?'
   - Initial data structure examination

#### 2. **Exploratory Data Analysis (EDA)**
   - Statistical summary
   - Missing value analysis
   - Class distribution analysis
   - Correlation analysis
   - Visualizations saved as `eda_visualization.png`

#### 3. **Data Cleaning & Preprocessing**
   - Missing value imputation using KNN Imputer (k=5)
   - Categorical variable encoding using Label Encoder
   - Converts all features to numeric format
   - Handles ~37% missing values in some features

#### 4. **Feature Engineering & Selection**
   - Feature importance calculation using Mutual Information
   - Identifies top predictive features
   - Visualization saved as `feature_importance.png`

#### 5. **Feature Scaling**
   - StandardScaler normalization (mean=0, std=1)
   - Essential for distance-based k-NN algorithm

#### 6. **Train-Test Split**
   - 80% training, 20% testing
   - Stratified split to maintain class distribution
   - 317 training samples, 80 test samples

#### 7. **Dimensionality Reduction (PCA)**
   - Reduces 24 features to 20 components
   - Retains 95% of variance
   - Comparison with/without PCA
   - Visualizations saved as `pca_analysis.png` and `pca_comparison.png`

#### 8. **k-NN Model Training**
   - Baseline model with k=5
   - Evaluation of k values from 1 to 30
   - Analysis of k parameter effects
   - Visualization saved as `k_value_analysis.png`

#### 9. **Hyperparameter Tuning**
   - Grid Search with 5-fold cross-validation
   - Parameters tuned:
     - n_neighbors: [3, 5, 7, 9, 11, 13, 15]
     - weights: ['uniform', 'distance']
     - metric: ['euclidean', 'manhattan', 'minkowski']
     - p: [1, 2]
   - Total 420 model configurations tested

#### 10. **Comprehensive Model Evaluation**
   - Multiple metrics: Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score
   - Cross-validation
   - Visualizations saved as `model_evaluation.png`

### Results

#### Best Model Performance
- **Test Accuracy**: 95.00%
- **Precision**: 95.59%
- **Recall**: 95.00%
- **F1-Score**: 95.05%
- **ROC-AUC**: 0.9787

#### Best Hyperparameters
- **n_neighbors**: 3
- **weights**: uniform
- **metric**: euclidean
- **p**: 1

#### Key Findings
1. **Optimal k value**: k=3-4 provides best performance
2. **PCA Impact**: 16.7% feature reduction with no accuracy loss
3. **Top Features**: Specific gravity (sg), Packed Cell Volume (pcv), Serum Creatinine (sc)
4. **Class Balance**: 60.08% (acceptable balance)

### Effect of k Parameter

The analysis thoroughly examines how the k parameter affects classification:

- **k = 1-3**: High accuracy but potential overfitting (high variance)
- **k = 4-7**: Optimal range - balanced bias-variance tradeoff
- **k = 8-15**: Slightly decreased accuracy
- **k > 15**: Underfitting risk (high bias)

**Optimal k = 3** achieved through:
1. Initial k-value sweep (1-30)
2. Grid search cross-validation
3. Test set validation

### Generated Files

After running the script, the following files are created:

1. **eda_visualization.png**: EDA plots (class distribution, missing values, correlation, age distribution)
2. **feature_importance.png**: Feature importance ranking
3. **pca_analysis.png**: PCA variance analysis
4. **k_value_analysis.png**: k parameter vs accuracy plot
5. **model_evaluation.png**: Confusion matrix, ROC curve, metrics comparison
6. **pca_comparison.png**: Performance with/without PCA
7. **classification_report.txt**: Comprehensive text report

### Project Structure
```
ML_Assignment/
│
├── data/
│   ├── chronic_kidney_disease.arff          # Main dataset
│   ├── chronic_kidney_disease_full.arff     # Full dataset
│   └── chronic_kidney_disease.info.txt      # Dataset documentation
│
├── knn_ckd_classification.py                # Main script
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
│
└── [Generated Files]
    ├── eda_visualization.png
    ├── feature_importance.png
    ├── pca_analysis.png
    ├── k_value_analysis.png
    ├── model_evaluation.png
    ├── pca_comparison.png
    └── classification_report.txt
```

### Code Features

#### Object-Oriented Design
- `CKDClassifier` class encapsulates entire pipeline
- Modular methods for each analysis step
- Easy to extend and customize

#### Robust Data Handling
- Manual ARFF parser for problematic files
- Handles missing values intelligently
- Automatic data type detection and conversion

#### Comprehensive Visualization
- Publication-quality plots
- Multiple visualization types
- Clear labeling and formatting

#### Reproducibility
- Fixed random seed (RANDOM_STATE=42)
- Complete documentation
- All steps logged to console

### Preprocessing Techniques Applied

1. **Missing Value Imputation**
   - KNN Imputer with 5 neighbors
   - Preserves data relationships
   - Handles both numeric and categorical

2. **Encoding**
   - Label Encoding for categorical variables
   - Binary encoding for target variable
   - Maintains all original feature information

3. **Scaling**
   - StandardScaler for feature normalization
   - Critical for distance-based k-NN
   - Mean=0, Standard Deviation=1

4. **Dimensionality Reduction**
   - PCA for feature reduction
   - 95% variance threshold
   - Reduces computational cost

### Model Validation

The model is validated using multiple approaches:

1. **Train-Test Split**: 80-20 stratified split
2. **Cross-Validation**: 5-fold CV on training data
3. **Grid Search**: Exhaustive hyperparameter search
4. **Multiple Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

### Assignment Requirements Checklist

✅ **k-NN Classifier Implementation**: Complete with multiple configurations
✅ **Data Preprocessing**: KNN imputation, encoding, scaling
✅ **Dimensionality Reduction**: PCA with variance analysis
✅ **k Parameter Analysis**: Comprehensive evaluation of k values (1-30)
✅ **Classification Accuracy**: 95%+ accuracy achieved
✅ **Visualizations**: 6 comprehensive visualization files
✅ **Documentation**: Complete code comments and README
✅ **Report**: Detailed text report generated

### Conclusion

This implementation demonstrates a complete machine learning pipeline for chronic kidney disease classification using k-NN. The model achieves excellent performance (95% accuracy) with proper preprocessing, feature engineering, and hyperparameter tuning. The analysis clearly shows the effect of the k parameter on classification accuracy and validates the effectiveness of dimensionality reduction techniques.

### References

- UCI Machine Learning Repository: Chronic Kidney Disease Dataset
- Scikit-learn Documentation: k-NN Classifier
- Python Data Science Handbook: Feature Engineering and Model Selection

### Author
CSCI 31022 - Machine Learning Assignment 1
Date: December 2024

### License
This project is for educational purposes as part of the CSCI 31022 course assignment.
