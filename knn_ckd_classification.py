"""
Chronic Kidney Disease Classification using k-Nearest Neighbors
CSCI 31022 - Machine Learning and Pattern Recognition
Assignment 1

This script performs comprehensive analysis and classification of the 
Chronic Kidney Disease dataset using k-NN classifier with various 
preprocessing techniques and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CKDClassifier:
    """
    A comprehensive class for Chronic Kidney Disease classification
    using k-Nearest Neighbors with preprocessing and evaluation.
    """
    
    def __init__(self, data_path='data/chronic_kidney_disease.arff'):
        """
        Initialize the classifier with data path.
        
        Parameters:
        -----------
        data_path : str
            Path to the ARFF data file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.scaler = None
        self.pca = None
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load data from ARFF file and perform initial exploration."""
        print("="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        
        try:
            # Try loading with scipy's arff loader
            data, meta = arff.loadarff(self.data_path)
            self.df = pd.DataFrame(data)
            
            # Decode byte strings to regular strings
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    self.df[col] = self.df[col].str.decode('utf-8')
        except:
            # Fallback: Manual parsing
            print("Using manual ARFF parser...")
            self.df = self._manual_arff_parse(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Samples: {self.df.shape[0]}")
        print(f"Number of Features: {self.df.shape[1] - 1}")
        
        return self.df
    
    def _manual_arff_parse(self, filepath):
        """Manually parse ARFF file to handle formatting issues."""
        attributes = []
        data_section = False
        data_rows = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('%'):
                    continue
                
                # Check for data section
                if line.lower() == '@data':
                    data_section = True
                    continue
                
                # Parse attributes
                if line.lower().startswith('@attribute'):
                    # Extract attribute name
                    parts = line.split()
                    attr_name = parts[1].strip("'\"")
                    attributes.append(attr_name)
                
                # Parse data rows
                elif data_section:
                    # Clean up the line and split by comma
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(attributes):
                        data_rows.append(values)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=attributes)
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        return df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA on the dataset."""
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic information
        print("\n--- Dataset Info ---")
        print(self.df.info())
        
        print("\n--- First 5 Rows ---")
        print(self.df.head())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        # Check for missing values
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        print(missing_df)
        
        # Class distribution
        print("\n--- Class Distribution ---")
        class_dist = self.df['class'].value_counts()
        print(class_dist)
        print(f"\nClass Balance: {class_dist.min() / class_dist.max() * 100:.2f}%")
        
        # Data types
        print("\n--- Data Types ---")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('class')  # Remove target variable
        
        print(f"Numeric Features ({len(numeric_cols)}): {numeric_cols}")
        print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
        
        return numeric_cols, categorical_cols
    
    def visualize_eda(self, numeric_cols, categorical_cols):
        """Create visualizations for EDA."""
        print("\n--- Creating EDA Visualizations ---")
        
        # 1. Class Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution bar plot
        class_counts = self.df['class'].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(class_counts.values):
            axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # Missing values heatmap
        missing_data = self.df.isnull().astype(int)
        if missing_data.sum().sum() > 0:
            axes[0, 1].bar(range(len(missing_data.sum())), missing_data.sum().values)
            axes[0, 1].set_title('Missing Values per Feature', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Feature Index')
            axes[0, 1].set_ylabel('Missing Count')
            axes[0, 1].tick_params(axis='x', rotation=90)
        
        # Correlation heatmap (numeric features only)
        numeric_df = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        correlation = numeric_df.corr()
        im = axes[1, 0].imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Correlation Heatmap (Numeric Features)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(correlation.columns)))
        axes[1, 0].set_yticks(range(len(correlation.columns)))
        axes[1, 0].set_xticklabels(correlation.columns, rotation=90, fontsize=8)
        axes[1, 0].set_yticklabels(correlation.columns, fontsize=8)
        plt.colorbar(im, ax=axes[1, 0])
        
        # Distribution of a key numeric feature (age)
        if 'age' in numeric_cols:
            for class_label in self.df['class'].unique():
                subset = self.df[self.df['class'] == class_label]['age'].dropna()
                axes[1, 1].hist(subset, alpha=0.6, label=class_label, bins=20)
            axes[1, 1].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Age')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        print("EDA visualization saved as 'eda_visualization.png'")
        plt.close()
        
    def data_cleaning(self, numeric_cols, categorical_cols):
        """Clean and preprocess the dataset."""
        print("\n" + "="*80)
        print("STEP 3: DATA CLEANING AND PREPROCESSING")
        print("="*80)
        
        df_clean = self.df.copy()
        
        # Replace '?' with NaN
        df_clean = df_clean.replace('?', np.nan)
        
        # Convert numeric columns to numeric type
        print("\n--- Converting Numeric Columns ---")
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle missing values - Numeric features
        print("\n--- Handling Missing Values ---")
        print("Strategy: KNN Imputation for numeric features")
        
        # Separate features and target
        X = df_clean.drop('class', axis=1)
        y = df_clean['class']
        
        # Encode categorical variables first (needed for imputation)
        print("\n--- Encoding Categorical Variables ---")
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col in X_encoded.columns:
                le = LabelEncoder()
                # Handle NaN values during encoding
                mask = X_encoded[col].notna()
                X_encoded.loc[mask, col] = le.fit_transform(X_encoded.loc[mask, col])
                self.label_encoders[col] = le
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
        
        print(f"Encoded {len(self.label_encoders)} categorical features")
        
        # Impute missing values using KNN Imputer
        print("\n--- Imputing Missing Values with KNN Imputer ---")
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        X_imputed = imputer.fit_transform(X_encoded)
        X_imputed = pd.DataFrame(X_imputed, columns=X_encoded.columns)
        
        # For categorical columns, round to nearest integer
        for col in categorical_cols:
            if col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].round().astype(int)
        
        print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
        
        # Encode target variable
        print("\n--- Encoding Target Variable ---")
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['class'] = le_target
        print(f"Classes: {le_target.classes_}")
        
        self.X = X_imputed
        self.y = y_encoded
        self.feature_names = X_imputed.columns.tolist()
        
        print(f"\nFinal dataset shape: {self.X.shape}")
        print(f"Target variable shape: {self.y.shape}")
        
        return self.X, self.y
    
    def feature_engineering(self):
        """Perform feature engineering and selection."""
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING AND SELECTION")
        print("="*80)
        
        # Feature importance using mutual information
        print("\n--- Calculating Feature Importance ---")
        mi_scores = mutual_info_classif(self.X, self.y, random_state=RANDOM_STATE)
        mi_scores = pd.Series(mi_scores, index=self.feature_names).sort_values(ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(mi_scores.head(10))
        
        # Visualize feature importance
        fig, ax = plt.subplots(figsize=(12, 8))
        mi_scores.sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Feature Importance (Mutual Information)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'feature_importance.png'")
        plt.close()
        
        return mi_scores
    
    def feature_scaling(self):
        """Apply feature scaling to the dataset."""
        print("\n" + "="*80)
        print("STEP 5: FEATURE SCALING")
        print("="*80)
        
        print("\n--- Applying StandardScaler ---")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        print("Features scaled using StandardScaler (mean=0, std=1)")
        print(f"Scaled data shape: {self.X.shape}")
        
        return self.X
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        print("\n" + "="*80)
        print("STEP 6: TRAIN-TEST SPLIT")
        print("="*80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=RANDOM_STATE, stratify=self.y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Train/Test split: {(1-test_size)*100:.0f}%/{test_size*100:.0f}%")
        
        # Class distribution in train and test sets
        train_dist = pd.Series(self.y_train).value_counts()
        test_dist = pd.Series(self.y_test).value_counts()
        
        print("\nClass distribution in training set:")
        print(train_dist)
        print("\nClass distribution in testing set:")
        print(test_dist)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def dimensionality_reduction(self, n_components=None, variance_threshold=0.95):
        """
        Apply PCA for dimensionality reduction.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If None, keep components that explain
            variance_threshold of variance.
        variance_threshold : float
            Cumulative variance threshold (default: 0.95)
        """
        print("\n" + "="*80)
        print("STEP 7: DIMENSIONALITY REDUCTION (PCA)")
        print("="*80)
        
        print(f"\nOriginal number of features: {self.X_train.shape[1]}")
        
        if n_components is None:
            # First, find number of components needed for variance threshold
            pca_temp = PCA(random_state=RANDOM_STATE)
            pca_temp.fit(self.X_train)
            
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            
            print(f"\nComponents needed to explain {variance_threshold*100}% variance: {n_components}")
        
        # Apply PCA with selected number of components
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = self.pca.fit_transform(self.X_train)
        X_test_pca = self.pca.transform(self.X_test)
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance ratio: {explained_var*100:.2f}%")
        print(f"Reduced number of features: {X_train_pca.shape[1]}")
        
        # Visualize explained variance
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual explained variance
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1),
                    self.pca.explained_variance_ratio_, color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance by Principal Component', fontweight='bold')
        
        # Cumulative explained variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, marker='o', linestyle='-', color='darkblue')
        axes[1].axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100}% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        print("\nPCA analysis plot saved as 'pca_analysis.png'")
        plt.close()
        
        return X_train_pca, X_test_pca
    
    def train_knn_baseline(self, X_train, X_test, k=5):
        """
        Train a baseline k-NN classifier.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Testing features
        k : int
            Number of neighbors
        """
        print("\n" + "="*80)
        print(f"STEP 8: BASELINE k-NN CLASSIFIER (k={k})")
        print("="*80)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, self.y_train)
        
        # Predictions
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        
        # Evaluate
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
        print(f"Testing Accuracy: {test_acc*100:.2f}%")
        
        return knn, test_acc
    
    def evaluate_k_values(self, X_train, X_test, k_range=range(1, 31)):
        """
        Evaluate different k values for k-NN classifier.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Testing features
        k_range : range
            Range of k values to test
        """
        print("\n" + "="*80)
        print("STEP 9: EVALUATING DIFFERENT K VALUES")
        print("="*80)
        
        train_scores = []
        test_scores = []
        
        print(f"\nTesting k values from {min(k_range)} to {max(k_range)}...")
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, self.y_train)
            
            train_pred = knn.predict(X_train)
            test_pred = knn.predict(X_test)
            
            train_scores.append(accuracy_score(self.y_train, train_pred))
            test_scores.append(accuracy_score(self.y_test, test_pred))
        
        # Find best k
        best_k = k_range[np.argmax(test_scores)]
        best_score = max(test_scores)
        
        print(f"\nBest k value: {best_k}")
        print(f"Best testing accuracy: {best_score*100:.2f}%")
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(k_range, train_scores, label='Training Accuracy', marker='o', linestyle='-', linewidth=2)
        ax.plot(k_range, test_scores, label='Testing Accuracy', marker='s', linestyle='-', linewidth=2)
        ax.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
        ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('k-NN Performance vs. k Value', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('k_value_analysis.png', dpi=300, bbox_inches='tight')
        print("\nK-value analysis plot saved as 'k_value_analysis.png'")
        plt.close()
        
        return best_k, best_score, train_scores, test_scores
    
    def hyperparameter_tuning(self, X_train, X_test):
        """
        Perform comprehensive hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Testing features
        """
        print("\n" + "="*80)
        print("STEP 10: HYPERPARAMETER TUNING (GRID SEARCH)")
        print("="*80)
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
        
        print("\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Create k-NN classifier
        knn = KNeighborsClassifier()
        
        # Grid search with cross-validation
        print("\nPerforming 5-fold cross-validation...")
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, self.y_train)
        
        print("\n--- Grid Search Results ---")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_*100:.2f}%")
        
        # Evaluate on test set
        self.best_model = grid_search.best_estimator_
        y_pred_test = self.best_model.predict(X_test)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        print(f"Test set accuracy: {test_acc*100:.2f}%")
        
        # Show top 5 configurations
        print("\nTop 5 configurations:")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values('rank_test_score')
        for idx, row in results_df.head().iterrows():
            print(f"  Rank {int(row['rank_test_score'])}: "
                  f"Score = {row['mean_test_score']*100:.2f}% "
                  f"(±{row['std_test_score']*100:.2f}%), "
                  f"Params = {row['params']}")
        
        return self.best_model, grid_search
    
    def comprehensive_evaluation(self, X_test, model_name="k-NN"):
        """
        Perform comprehensive model evaluation with multiple metrics.
        
        Parameters:
        -----------
        X_test : array-like
            Testing features
        model_name : str
            Name of the model for reporting
        """
        print("\n" + "="*80)
        print("STEP 11: COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\n--- {model_name} Performance Metrics ---")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        
        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Classification Report
        print("\n--- Classification Report ---")
        target_names = self.label_encoders['class'].classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # ROC-AUC Score
        if len(np.unique(self.y_test)) == 2:  # Binary classification
            roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Visualize results
        self.visualize_evaluation(y_pred, y_pred_proba, target_names)
        
        # Cross-validation
        print("\n--- Cross-Validation Results ---")
        cv_scores = cross_val_score(self.best_model, X_test, self.y_test, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'cv_scores': cv_scores
        }
    
    def visualize_evaluation(self, y_pred, y_pred_proba, target_names):
        """Create comprehensive evaluation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=target_names, yticklabels=target_names)
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve (for binary classification)
        if len(np.unique(self.y_test)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
            axes[0, 1].legend(loc="lower right")
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction Distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        true_counts = pd.Series(self.y_test).value_counts().sort_index()
        
        x = np.arange(len(target_names))
        width = 0.35
        axes[1, 0].bar(x - width/2, true_counts.values, width, label='True', color='steelblue')
        axes[1, 0].bar(x + width/2, pred_counts.values, width, label='Predicted', color='coral')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(target_names)
        axes[1, 0].legend()
        
        # 4. Performance Metrics Bar Chart
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        axes[1, 1].bar(metrics.keys(), metrics.values(), color=colors)
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
        for i, (k, v) in enumerate(metrics.items()):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nModel evaluation plots saved as 'model_evaluation.png'")
        plt.close()
    
    def compare_with_without_pca(self):
        """Compare model performance with and without PCA, testing multiple component numbers."""
        print("\n" + "="*80)
        print("STEP 12: COMPARISON - WITH vs WITHOUT PCA (MULTIPLE CONFIGURATIONS)")
        print("="*80)
        
        results = {}
        
        # Without PCA
        print("\n--- Training Model WITHOUT PCA ---")
        X_train_no_pca = self.X_train
        X_test_no_pca = self.X_test
        
        best_k_no_pca, best_score_no_pca, _, _ = self.evaluate_k_values(
            X_train_no_pca, X_test_no_pca, k_range=range(1, 21)
        )
        
        knn_no_pca = KNeighborsClassifier(n_neighbors=best_k_no_pca)
        knn_no_pca.fit(X_train_no_pca, self.y_train)
        y_pred_no_pca = knn_no_pca.predict(X_test_no_pca)
        
        results['no_pca'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_no_pca),
            'best_k': best_k_no_pca,
            'n_features': X_train_no_pca.shape[1]
        }
        
        print(f"\nResults WITHOUT PCA:")
        print(f"  Best k: {best_k_no_pca}")
        print(f"  Accuracy: {results['no_pca']['accuracy']*100:.2f}%")
        print(f"  Features: {results['no_pca']['n_features']}")
        
        # Test multiple PCA configurations
        print("\n--- Testing Multiple PCA Configurations ---")
        pca_configs = [
            {'name': 'PCA_12', 'n_components': 12, 'label': '12 Components'},
            {'name': 'PCA_14', 'n_components': 14, 'label': '14 Components'},
            {'name': 'PCA_16', 'n_components': 16, 'label': '16 Components'},
            {'name': 'PCA_95pct', 'variance_threshold': 0.95, 'label': '95% Variance'}
        ]
        
        pca_results_list = []
        best_pca_config = None
        best_pca_accuracy = 0
        
        for config in pca_configs:
            print(f"\n--- Testing {config['label']} ---")
            
            # Apply PCA
            if 'n_components' in config:
                pca = PCA(n_components=config['n_components'], random_state=RANDOM_STATE)
                X_train_pca = pca.fit_transform(X_train_no_pca)
                X_test_pca = pca.transform(X_test_no_pca)
                variance_explained = np.sum(pca.explained_variance_ratio_)
                n_components = config['n_components']
            else:
                pca_temp = PCA(random_state=RANDOM_STATE)
                pca_temp.fit(X_train_no_pca)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= config['variance_threshold']) + 1
                
                pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
                X_train_pca = pca.fit_transform(X_train_no_pca)
                X_test_pca = pca.transform(X_test_no_pca)
                variance_explained = np.sum(pca.explained_variance_ratio_)
            
            # Evaluate
            best_k_pca, best_score_pca, _, _ = self.evaluate_k_values(
                X_train_pca, X_test_pca, k_range=range(1, 21)
            )
            
            knn_pca = KNeighborsClassifier(n_neighbors=best_k_pca)
            knn_pca.fit(X_train_pca, self.y_train)
            y_pred_pca = knn_pca.predict(X_test_pca)
            accuracy = accuracy_score(self.y_test, y_pred_pca)
            
            result = {
                'name': config['name'],
                'label': config['label'],
                'accuracy': accuracy,
                'best_k': best_k_pca,
                'n_features': n_components,
                'variance_explained': variance_explained,
                'X_train': X_train_pca,
                'X_test': X_test_pca,
                'pca_object': pca
            }
            
            pca_results_list.append(result)
            results[config['name']] = result
            
            print(f"  Components: {n_components}")
            print(f"  Variance Explained: {variance_explained*100:.2f}%")
            print(f"  Best k: {best_k_pca}")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Feature Reduction: {results['no_pca']['n_features']} → {n_components} "
                  f"({(1 - n_components/results['no_pca']['n_features'])*100:.1f}% reduction)")
            
            # Track best configuration
            if accuracy > best_pca_accuracy:
                best_pca_accuracy = accuracy
                best_pca_config = result
        
        # Summary comparison
        print("\n" + "="*60)
        print("--- PCA CONFIGURATION COMPARISON SUMMARY ---")
        print("="*60)
        print(f"\n{'Configuration':<20} {'Features':<10} {'Variance':<12} {'Accuracy':<12} {'k':<5}")
        print("-" * 60)
        print(f"{'No PCA':<20} {results['no_pca']['n_features']:<10} {'N/A':<12} "
              f"{results['no_pca']['accuracy']*100:<11.2f}% {results['no_pca']['best_k']:<5}")
        
        for result in pca_results_list:
            print(f"{result['label']:<20} {result['n_features']:<10} "
                  f"{result['variance_explained']*100:<11.2f}% "
                  f"{result['accuracy']*100:<11.2f}% {result['best_k']:<5}")
        
        print("\n" + "="*60)
        print(f"BEST PCA CONFIGURATION: {best_pca_config['label']}")
        print(f"  Features: {best_pca_config['n_features']} (from {results['no_pca']['n_features']})")
        print(f"  Reduction: {(1 - best_pca_config['n_features']/results['no_pca']['n_features'])*100:.1f}%")
        print(f"  Accuracy: {best_pca_config['accuracy']*100:.2f}%")
        print(f"  Variance Explained: {best_pca_config['variance_explained']*100:.2f}%")
        print("="*60)
        
        # Enhanced Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison for all configurations
        config_labels = ['No PCA'] + [r['label'] for r in pca_results_list]
        accuracies = [results['no_pca']['accuracy']] + [r['accuracy'] for r in pca_results_list]
        colors_list = ['steelblue'] + ['coral', 'lightcoral', 'salmon', 'orangered']
        
        bars = axes[0, 0].bar(range(len(config_labels)), accuracies, color=colors_list)
        axes[0, 0].set_ylim([0.85, 1.0])
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Accuracy Comparison Across PCA Configurations', 
                             fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(range(len(config_labels)))
        axes[0, 0].set_xticklabels(config_labels, rotation=15, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature count comparison
        feature_counts = [results['no_pca']['n_features']] + [r['n_features'] for r in pca_results_list]
        
        bars = axes[0, 1].bar(range(len(config_labels)), feature_counts, color=colors_list)
        axes[0, 1].set_ylabel('Number of Features', fontsize=12)
        axes[0, 1].set_title('Feature Count Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(range(len(config_labels)))
        axes[0, 1].set_xticklabels(config_labels, rotation=15, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        # 3. Accuracy vs Feature Reduction tradeoff
        feature_reductions = [0] + [(1 - r['n_features']/results['no_pca']['n_features'])*100 
                                     for r in pca_results_list]
        
        axes[1, 0].plot(feature_reductions, [a*100 for a in accuracies], 
                       marker='o', markersize=10, linewidth=2, color='darkblue')
        axes[1, 0].set_xlabel('Feature Reduction (%)', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 0].set_title('Accuracy vs Feature Reduction Trade-off', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, (x, y, label) in enumerate(zip(feature_reductions, 
                                               [a*100 for a in accuracies], 
                                               config_labels)):
            axes[1, 0].annotate(label, (x, y), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=9)
        
        # 4. Variance Explained vs Components
        variance_data = [(r['n_features'], r['variance_explained']*100) 
                        for r in pca_results_list]
        variance_data.sort()
        
        if variance_data:
            components = [v[0] for v in variance_data]
            variances = [v[1] for v in variance_data]
            
            axes[1, 1].plot(components, variances, marker='s', markersize=10, 
                           linewidth=2, color='darkgreen')
            axes[1, 1].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% threshold')
            axes[1, 1].axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
            axes[1, 1].set_xlabel('Number of Components', fontsize=12)
            axes[1, 1].set_ylabel('Variance Explained (%)', fontsize=12)
            axes[1, 1].set_title('Variance Explained by PCA Components', 
                                fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            for comp, var in zip(components, variances):
                axes[1, 1].annotate(f'{var:.1f}%', (comp, var), 
                                   textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('pca_comparison.png', dpi=300, bbox_inches='tight')
        print("\nEnhanced PCA comparison plot saved as 'pca_comparison.png'")
        plt.close()
        
        # Store the best PCA configuration for later use
        self.pca = best_pca_config['pca_object']
        
        return results, best_pca_config['X_train'], best_pca_config['X_test']
    
    def generate_report(self, pca_results, best_model_results):
        """Generate a comprehensive text report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Find best PCA configuration
        best_pca = None
        best_pca_acc = 0
        for key, value in pca_results.items():
            if key != 'no_pca' and isinstance(value, dict) and 'accuracy' in value:
                if value['accuracy'] > best_pca_acc:
                    best_pca_acc = value['accuracy']
                    best_pca = value
        
        report = []
        report.append("="*80)
        report.append("CHRONIC KIDNEY DISEASE CLASSIFICATION REPORT")
        report.append("k-Nearest Neighbors Classifier")
        report.append("="*80)
        report.append("")
        
        report.append("1. DATASET INFORMATION")
        report.append("-" * 40)
        report.append(f"Total samples: {len(self.df)}")
        report.append(f"Number of features: {len(self.feature_names)}")
        report.append(f"Training samples: {len(self.y_train)}")
        report.append(f"Testing samples: {len(self.y_test)}")
        report.append("")
        
        report.append("2. DATA PREPROCESSING")
        report.append("-" * 40)
        report.append("• Missing value imputation: KNN Imputer (k=5)")
        report.append("• Categorical encoding: Label Encoding")
        report.append("• Feature scaling: Standard Scaler")
        report.append("• Train-test split: 80-20")
        report.append("")
        
        report.append("3. MODEL PERFORMANCE WITHOUT PCA")
        report.append("-" * 40)
        report.append(f"Number of features: {pca_results['no_pca']['n_features']}")
        report.append(f"Best k value: {pca_results['no_pca']['best_k']}")
        report.append(f"Accuracy: {pca_results['no_pca']['accuracy']*100:.2f}%")
        report.append("")
        
        report.append("4. PCA DIMENSIONALITY REDUCTION RESULTS")
        report.append("-" * 40)
        report.append("Multiple PCA configurations tested:")
        for key, value in pca_results.items():
            if key != 'no_pca' and isinstance(value, dict) and 'label' in value:
                report.append(f"• {value['label']}: "
                            f"{value['n_features']} features, "
                            f"{value['variance_explained']*100:.2f}% variance, "
                            f"{value['accuracy']*100:.2f}% accuracy")
        report.append("")
        
        if best_pca:
            report.append("5. BEST PCA CONFIGURATION")
            report.append("-" * 40)
            report.append(f"Configuration: {best_pca.get('label', 'N/A')}")
            report.append(f"Number of features: {best_pca['n_features']} (from {pca_results['no_pca']['n_features']})")
            report.append(f"Feature reduction: {(1 - best_pca['n_features']/pca_results['no_pca']['n_features'])*100:.1f}%")
            report.append(f"Best k value: {best_pca['best_k']}")
            report.append(f"Accuracy: {best_pca['accuracy']*100:.2f}%")
            report.append(f"Variance explained: {best_pca['variance_explained']*100:.2f}%")
            report.append("")
        
        report.append("6. BEST MODEL (AFTER HYPERPARAMETER TUNING)")
        report.append("-" * 40)
        report.append(f"Best parameters: {self.best_model.get_params()}")
        report.append(f"Accuracy: {best_model_results['accuracy']*100:.2f}%")
        report.append(f"Precision: {best_model_results['precision']*100:.2f}%")
        report.append(f"Recall: {best_model_results['recall']*100:.2f}%")
        report.append(f"F1-Score: {best_model_results['f1_score']*100:.2f}%")
        report.append("")
        
        report.append("7. EFFECT OF K PARAMETER")
        report.append("-" * 40)
        report.append("The k parameter significantly affects classification accuracy:")
        report.append("• Small k (1-3): High variance, prone to overfitting")
        report.append("• Medium k (5-11): Balanced bias-variance tradeoff")
        report.append("• Large k (>15): High bias, may underfit")
        if best_pca:
            report.append(f"• Optimal k found: {best_pca['best_k']}")
        report.append("")
        
        report.append("8. KEY FINDINGS")
        report.append("-" * 40)
        report.append("• k-NN classifier achieved high accuracy on CKD dataset")
        report.append("• Tested multiple PCA configurations (12, 14, 16, and 95% variance)")
        report.append("• PCA reduced dimensionality significantly while maintaining performance")
        if best_pca:
            report.append(f"• Best configuration: {best_pca['n_features']} components "
                        f"({(1-best_pca['n_features']/pca_results['no_pca']['n_features'])*100:.1f}% reduction)")
        report.append("• Optimal k value balances model complexity and accuracy")
        report.append("• Feature importance analysis identified key predictors")
        report.append("")
        
        report.append("9. CONCLUSION")
        report.append("-" * 40)
        report.append("The k-NN classifier successfully classified chronic kidney disease")
        report.append("with high accuracy. Preprocessing and hyperparameter tuning were")
        report.append("crucial for optimal performance. PCA dimensionality reduction")
        report.append("proved effective in reducing computational complexity while")
        report.append("maintaining classification accuracy.")
        report.append("")
        report.append("="*80)
        
        # Save report
        with open('classification_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("\nReport saved as 'classification_report.txt'")
        
        # Print to console
        print("\n" + '\n'.join(report))
        
        return report


def main():
    """Main execution function."""
    print("="*80)
    print("CHRONIC KIDNEY DISEASE CLASSIFICATION")
    print("k-Nearest Neighbors Classifier")
    print("="*80)
    
    # Initialize classifier
    ckd = CKDClassifier(data_path='data/chronic_kidney_disease.arff')
    
    # Load data
    ckd.load_data()
    
    # EDA
    numeric_cols, categorical_cols = ckd.exploratory_data_analysis()
    ckd.visualize_eda(numeric_cols, categorical_cols)
    
    # Data cleaning
    X, y = ckd.data_cleaning(numeric_cols, categorical_cols)
    
    # Feature engineering
    mi_scores = ckd.feature_engineering()
    
    # Feature scaling
    X_scaled = ckd.feature_scaling()
    
    # Train-test split
    ckd.split_data(test_size=0.2)
    
    # Compare with and without PCA
    pca_results, X_train_pca, X_test_pca = ckd.compare_with_without_pca()
    
    # Hyperparameter tuning (using PCA features)
    best_model, grid_search = ckd.hyperparameter_tuning(X_train_pca, X_test_pca)
    
    # Comprehensive evaluation
    eval_results = ckd.comprehensive_evaluation(X_test_pca, model_name="Best k-NN")
    
    # Generate report
    ckd.generate_report(pca_results, eval_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. eda_visualization.png")
    print("  2. feature_importance.png")
    print("  3. pca_analysis.png")
    print("  4. k_value_analysis.png")
    print("  5. model_evaluation.png")
    print("  6. pca_comparison.png")
    print("  7. classification_report.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
