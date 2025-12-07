"""
Chronic Kidney Disease Classification using k-Nearest Neighbors
Enhanced version combining comprehensive analysis with optimal PCA implementation

This script performs comprehensive analysis and classification of the 
Chronic Kidney Disease dataset using k-NN classifier with various 
preprocessing techniques and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from pathlib import Path
import warnings
import json
import joblib
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
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
    Enhanced class for Chronic Kidney Disease classification
    using k-Nearest Neighbors with comprehensive preprocessing and evaluation.
    """
    
    def __init__(self, data_path='data/chronic_kidney_disease.arff'):
        """Initialize the classifier with data path."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.preprocessor = None
        self.best_model = None
        self.feature_names = None
        self.numeric_cols = []
        self.categorical_cols = []
        
    def load_arff(self, path):
        """Robust ARFF file loader with fallback parser."""
        try:
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            # Decode byte strings
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            return df
        except Exception:
            # Manual parser fallback
            attrs = []
            data_lines = []
            in_data = False
            txt = Path(path).read_text(encoding='utf-8', errors='ignore')
            for raw in txt.splitlines():
                line = raw.strip()
                if not line or line.startswith('%'):
                    continue
                low = line.lower()
                if low.startswith('@attribute'):
                    parts = line.split()
                    if len(parts) >= 2:
                        attrs.append(parts[1].strip("'\""))
                elif low.startswith('@data'):
                    in_data = True
                    continue
                elif in_data:
                    parts = [p.strip().strip("'\"") for p in line.split(',')]
                    if len(parts) < len(attrs):
                        parts += [None] * (len(attrs) - len(parts))
                    elif len(parts) > len(attrs):
                        parts = parts[:len(attrs)]
                    data_lines.append(parts)
            return pd.DataFrame(data_lines, columns=attrs)
    
    def load_data(self):
        """Load data from ARFF file and perform initial exploration."""
        print("="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        
        self.df = self.load_arff(self.data_path)
        self.df = self.df.replace('?', np.nan)
        
        # Normalize column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        
        # Normalize string columns
        for c in self.df.select_dtypes(include=['object']).columns:
            self.df[c] = self.df[c].astype(str).str.strip().str.lower().replace({'nan': None})
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Samples: {self.df.shape[0]}")
        print(f"Number of Features: {self.df.shape[1] - 1}")
        
        return self.df
    
    def prepare_target(self):
        """Prepare and normalize target variable."""
        # Normalize target and map
        self.df['class'] = self.df['class'].astype(str).str.strip().str.lower()
        # Fix variants
        self.df['class'] = self.df['class'].replace({
            'no': 'notckd', 
            'not ckd': 'notckd', 
            'not_ckd': 'notckd'
        })
        mapped = self.df['class'].map({'ckd': 1, 'notckd': 0})
        n_unmapped = mapped.isna().sum()
        if n_unmapped > 0:
            print(f"\nWarning: Dropping {n_unmapped} rows with unmapped class values")
            self.df = self.df.loc[mapped.notna()].copy()
            mapped = mapped.loc[mapped.notna()]
        self.df['class'] = mapped.astype(int)
    
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
        
        # Define numeric and categorical columns
        numeric_candidates = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
        self.numeric_cols = [c for c in numeric_candidates if c in self.df.columns]
        self.categorical_cols = [c for c in self.df.columns if c not in self.numeric_cols + ['class']]
        
        print("\n--- Data Types ---")
        print(f"Numeric Features ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"Categorical Features ({len(self.categorical_cols)}): {self.categorical_cols}")
        
        return self.numeric_cols, self.categorical_cols
    
    def visualize_eda(self):
        """Create visualizations for EDA."""
        print("\n--- Creating EDA Visualizations ---")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Class Distribution
        class_counts = self.df['class'].value_counts()
        axes[0, 0].bar(['Not CKD', 'CKD'], class_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(class_counts.values):
            axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # 2. Missing values bar plot
        missing_data = self.df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            missing_data.plot(kind='barh', ax=axes[0, 1], color='coral')
            axes[0, 1].set_title('Missing Values per Feature', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Missing Count')
        
        # 3. Correlation heatmap (numeric features only)
        numeric_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
        correlation = numeric_df.corr()
        im = axes[1, 0].imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Correlation Heatmap (Numeric Features)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(correlation.columns)))
        axes[1, 0].set_yticks(range(len(correlation.columns)))
        axes[1, 0].set_xticklabels(correlation.columns, rotation=90, fontsize=8)
        axes[1, 0].set_yticklabels(correlation.columns, fontsize=8)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Age distribution by class
        if 'age' in self.numeric_cols:
            for class_label in [0, 1]:
                subset = self.df[self.df['class'] == class_label]['age'].apply(pd.to_numeric, errors='coerce').dropna()
                label = 'Not CKD' if class_label == 0 else 'CKD'
                axes[1, 1].hist(subset, alpha=0.6, label=label, bins=20)
            axes[1, 1].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Age')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        print("EDA visualization saved as 'eda_visualization.png'")
        plt.close()
    
    def data_cleaning(self):
        """Clean and preprocess the dataset."""
        print("\n" + "="*80)
        print("STEP 3: DATA CLEANING AND PREPROCESSING")
        print("="*80)
        
        # Coerce numeric columns
        print("\n--- Converting Numeric Columns ---")
        for c in self.numeric_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        
        # Create derived feature (age group)
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['age'].astype(float), 
                bins=[0,18,35,50,65,200],
                labels=['child','young','adult','mid','senior']
            )
            if 'age_group' not in self.categorical_cols:
                self.categorical_cols.append('age_group')
            print("Created derived feature: age_group")
        
        # Split X, y
        X = self.df.drop(columns=['class']).reset_index(drop=True)
        y = self.df['class'].reset_index(drop=True)
        
        # Label-encode categorical columns for KNN imputation
        print("\n--- Encoding Categorical Variables ---")
        X_enc = X.copy()
        
        # Convert categorical columns to object type
        for col in self.categorical_cols:
            if col in X_enc.columns and X_enc[col].dtype.name == 'category':
                X_enc[col] = X_enc[col].astype(object)
        
        for col in self.categorical_cols:
            if col not in X_enc.columns:
                continue
            le = LabelEncoder()
            mask = X_enc[col].notna()
            if mask.sum() > 0:
                X_enc.loc[mask, col] = le.fit_transform(X_enc.loc[mask, col])
                self.label_encoders[col] = le
            else:
                X_enc[col] = np.nan
            X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce')
        
        print(f"Encoded {len(self.label_encoders)} categorical features")
        
        # KNN Imputation
        print("\n--- Imputing Missing Values with KNN Imputer ---")
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)
        
        # Round categorical columns to integers
        for col in self.categorical_cols:
            if col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].round().astype(int)
        
        print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
        
        self.X = X_imputed
        self.y = y
        self.feature_names = X_imputed.columns.tolist()
        
        print(f"\nFinal dataset shape: {self.X.shape}")
        print(f"Target variable shape: {self.y.shape}")
        
        return self.X, self.y
    
    def feature_engineering(self):
        """Perform feature importance analysis."""
        print("\n" + "="*80)
        print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Calculate mutual information scores
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
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        print("\n" + "="*80)
        print("STEP 5: TRAIN-TEST SPLIT")
        print("="*80)
        
        # Shuffle data
        self.X, self.y = shuffle(self.X, self.y, random_state=RANDOM_STATE)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=RANDOM_STATE, stratify=self.y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Train/Test split: {(1-test_size)*100:.0f}%/{test_size*100:.0f}%")
        
        # Class distribution
        train_dist = pd.Series(self.y_train).value_counts()
        test_dist = pd.Series(self.y_test).value_counts()
        
        print("\nClass distribution in training set:")
        print(train_dist)
        print("\nClass distribution in testing set:")
        print(test_dist)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_pipeline(self):
        """Build preprocessing and modeling pipeline with optimal PCA."""
        print("\n" + "="*80)
        print("STEP 6: BUILDING PREPROCESSING PIPELINE")
        print("="*80)
        
        # Get feature lists
        num_feats = [c for c in self.numeric_cols if c in self.X.columns]
        cat_feats = [c for c in self.categorical_cols if c in self.X.columns]
        
        print(f"\nNumeric features for scaling: {len(num_feats)}")
        print(f"Categorical features for encoding: {len(cat_feats)}")
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
        ], remainder='drop')
        
        # Build pipeline with PCA and KNN
        pipeline = Pipeline([
            ('pre', self.preprocessor),
            ('pca', PCA(random_state=RANDOM_STATE)),
            ('clf', KNeighborsClassifier())
        ])
        
        print("\nPipeline created: Preprocessor -> PCA -> k-NN")
        
        return pipeline
    
    def hyperparameter_tuning(self, pipeline):
        """Perform comprehensive hyperparameter tuning using GridSearchCV."""
        print("\n" + "="*80)
        print("STEP 7: HYPERPARAMETER TUNING (GRID SEARCH)")
        print("="*80)
        
        # Estimate encoded feature count for PCA options
        sample_encoded = self.preprocessor.fit_transform(self.X_train.iloc[:10])
        encoded_dim = sample_encoded.shape[1]
        print(f"\nEncoded feature dimension: {encoded_dim}")
        
        # Create smart PCA component choices
        n_comp_choices = sorted(list({
            min(5, encoded_dim), 
            min(8, encoded_dim), 
            min(10, encoded_dim),
            min(12, encoded_dim), 
            min(15, encoded_dim), 
            encoded_dim
        }))
        n_comp_choices = [int(c) for c in n_comp_choices if c >= 1]
        
        print(f"PCA component choices: {n_comp_choices}")
        
        # Define parameter grid
        param_grid = {
            'pca__n_components': n_comp_choices,
            'clf__n_neighbors': [1, 3, 4, 5, 7, 9, 11],
            'clf__weights': ['uniform', 'distance'],
            'clf__p': [1, 2]
        }
        
        print("\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Grid search with stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nPerforming 5-fold stratified cross-validation...")
        grid_search.fit(self.X_train, self.y_train)
        
        print("\n--- Grid Search Results ---")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        # Show top 5 configurations
        print("\nTop 5 configurations:")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values('rank_test_score')
        for idx, row in results_df.head().iterrows():
            print(f"  Rank {int(row['rank_test_score'])}: "
                  f"ROC-AUC = {row['mean_test_score']:.4f} "
                  f"(±{row['std_test_score']:.4f}), "
                  f"Params = {row['params']}")
        
        return self.best_model, grid_search
    
    def analyze_pca(self):
        """Analyze PCA components from best model."""
        print("\n" + "="*80)
        print("STEP 8: PCA ANALYSIS")
        print("="*80)
        
        # Extract PCA from pipeline
        pca = self.best_model.named_steps['pca']
        n_components = pca.n_components_
        explained_var = np.sum(pca.explained_variance_ratio_)
        
        print(f"\nNumber of PCA components: {n_components}")
        print(f"Total explained variance: {explained_var*100:.2f}%")
        print(f"\nExplained variance per component:")
        for i, var in enumerate(pca.explained_variance_ratio_, 1):
            print(f"  PC{i}: {var*100:.2f}%")
        
        # Visualize PCA analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual explained variance
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                    pca.explained_variance_ratio_, color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance by Principal Component', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, marker='o', linestyle='-', color='darkblue')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        print("\nPCA analysis plot saved as 'pca_analysis.png'")
        plt.close()
    
    def evaluate_k_values(self):
        """Evaluate different k values to visualize performance."""
        print("\n" + "="*80)
        print("STEP 9: K-VALUE ANALYSIS")
        print("="*80)
        
        # Transform data using best model's preprocessor and PCA
        X_train_transformed = self.best_model.named_steps['pre'].transform(self.X_train)
        X_train_transformed = self.best_model.named_steps['pca'].transform(X_train_transformed)
        X_test_transformed = self.best_model.named_steps['pre'].transform(self.X_test)
        X_test_transformed = self.best_model.named_steps['pca'].transform(X_test_transformed)
        
        k_range = range(1, 31)
        train_scores = []
        test_scores = []
        
        print(f"\nTesting k values from {min(k_range)} to {max(k_range)}...")
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_transformed, self.y_train)
            
            train_pred = knn.predict(X_train_transformed)
            test_pred = knn.predict(X_test_transformed)
            
            train_scores.append(accuracy_score(self.y_train, train_pred))
            test_scores.append(accuracy_score(self.y_test, test_pred))
        
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
    
    def comprehensive_evaluation(self):
        """Perform comprehensive model evaluation with multiple metrics."""
        print("\n" + "="*80)
        print("STEP 10: COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n--- Model Performance Metrics ---")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Classification Report
        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, y_pred, target_names=['Not CKD', 'CKD']))
        
        # Cross-validation
        print("\n--- Cross-Validation Results ---")
        cv_scores = cross_val_score(self.best_model, self.X_test, self.y_test, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
        
        # Visualize results
        self.visualize_evaluation(y_pred, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'cv_scores': cv_scores
        }
    
    def visualize_evaluation(self, y_pred, y_pred_proba):
        """Create comprehensive evaluation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
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
        
        x = np.arange(2)
        width = 0.35
        axes[1, 0].bar(x - width/2, true_counts.values, width, label='True', color='steelblue')
        axes[1, 0].bar(x + width/2, pred_counts.values, width, label='Predicted', color='coral')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Not CKD', 'CKD'])
        axes[1, 0].legend()
        
        # 4. Performance Metrics Bar Chart
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred)
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
    
    def save_model(self):
        """Save the trained model and preprocessing information."""
        print("\n" + "="*80)
        print("STEP 11: SAVING MODEL")
        print("="*80)
        
        # Save model
        joblib.dump(self.best_model, 'knn_ckd_final_model.joblib')
        print("\nModel saved as 'knn_ckd_final_model.joblib'")
        
        # Save preprocessing info
        preprocess_info = {
            'label_encoders': self.label_encoders,
            'numeric_features': self.numeric_cols,
            'categorical_features': self.categorical_cols
        }
        joblib.dump(preprocess_info, 'ckd_preprocess_info.joblib')
        print("Preprocessing info saved as 'ckd_preprocess_info.joblib'")
    
    def generate_report(self, eval_results, grid_search):
        """Generate a comprehensive text report."""
        print("\n" + "="*80)
        print("STEP 12: GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Extract PCA info
        pca = self.best_model.named_steps['pca']
        n_components = pca.n_components_
        explained_var = np.sum(pca.explained_variance_ratio_)
        
        # Extract only serializable best parameters
        best_params = grid_search.best_params_
        serializable_params = {}
        for key, value in best_params.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_params[key] = value
            else:
                serializable_params[key] = str(value)
        
        # Create summary dictionary
        summary = {
            'best_params': serializable_params,
            'cv_best_score': float(grid_search.best_score_),
            'n_pca_components': int(n_components),
            'pca_explained_variance': float(explained_var),
            'test_metrics': {
                'accuracy': float(eval_results['accuracy']),
                'precision': float(eval_results['precision']),
                'recall': float(eval_results['recall']),
                'f1_score': float(eval_results['f1_score']),
                'roc_auc': float(eval_results['roc_auc'])
            }
        }
        
        # Save as JSON
        with open('ckd_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("\nSummary saved as 'ckd_summary.json'")
        
        # Create text report
        report = []
        report.append("="*80)
        report.append("CHRONIC KIDNEY DISEASE CLASSIFICATION REPORT")
        report.append("k-Nearest Neighbors Classifier with Optimal PCA")
        report.append("="*80)
        report.append("")
        
        report.append("1. DATASET INFORMATION")
        report.append("-" * 40)
        report.append(f"Total samples: {len(self.df)}")
        report.append(f"Number of original features: {len(self.feature_names)}")
        report.append(f"Training samples: {len(self.y_train)}")
        report.append(f"Testing samples: {len(self.y_test)}")
        report.append("")
        
        report.append("2. DATA PREPROCESSING")
        report.append("-" * 40)
        report.append("• Missing value imputation: KNN Imputer (k=5)")
        report.append("• Categorical encoding: Label Encoding + One-Hot Encoding")
        report.append("• Feature scaling: Standard Scaler (for numeric features)")
        report.append("• Train-test split: 80-20 (stratified)")
        report.append(f"• Derived features: age_group")
        report.append("")
        
        report.append("3. DIMENSIONALITY REDUCTION")
        report.append("-" * 40)
        report.append(f"• PCA components: {n_components}")
        report.append(f"• Explained variance: {explained_var*100:.2f}%")
        report.append("• Strategy: Optimal component selection via grid search")
        report.append("")
        
        report.append("4. BEST MODEL HYPERPARAMETERS")
        report.append("-" * 40)
        best_params = grid_search.best_params_
        for param, value in best_params.items():
            report.append(f"• {param}: {value}")
        report.append("")
        
        report.append("5. MODEL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")
        report.append(f"Test Set Accuracy: {eval_results['accuracy']*100:.2f}%")
        report.append(f"Test Set Precision: {eval_results['precision']*100:.2f}%")
        report.append(f"Test Set Recall: {eval_results['recall']*100:.2f}%")
        report.append(f"Test Set F1-Score: {eval_results['f1_score']*100:.2f}%")
        report.append(f"Test Set ROC-AUC: {eval_results['roc_auc']:.4f}")
        report.append("")
        
        report.append("6. CONFUSION MATRIX")
        report.append("-" * 40)
        cm = eval_results['confusion_matrix']
        report.append(f"True Negatives:  {cm[0, 0]}")
        report.append(f"False Positives: {cm[0, 1]}")
        report.append(f"False Negatives: {cm[1, 0]}")
        report.append(f"True Positives:  {cm[1, 1]}")
        report.append("")
        
        report.append("7. KEY FINDINGS")
        report.append("-" * 40)
        report.append(f"• PCA reduced features while maintaining {explained_var*100:.1f}% variance")
        report.append(f"• Optimal k value: {best_params.get('clf__n_neighbors', 'N/A')}")
        report.append(f"• Distance weighting: {best_params.get('clf__weights', 'N/A')}")
        report.append(f"• Distance metric: p={best_params.get('clf__p', 'N/A')} (Minkowski)")
        report.append(f"• Model achieved {eval_results['roc_auc']:.4f} ROC-AUC score")
        report.append("")
        
        report.append("8. CONCLUSION")
        report.append("-" * 40)
        report.append("The optimized k-NN classifier with PCA successfully classified")
        report.append("chronic kidney disease with high accuracy and efficiency.")
        report.append(f"Using only {n_components} PCA components, the model achieved")
        report.append(f"{eval_results['accuracy']*100:.1f}% accuracy, demonstrating effective")
        report.append("dimensionality reduction without sacrificing performance.")
        report.append("")
        report.append("="*80)
        
        # Save report
        with open('classification_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("\nReport saved as 'classification_report.txt'")
        print("\n" + '\n'.join(report))


def main(data_path='data/chronic_kidney_disease.arff'):
    """Main execution function."""
    print("="*80)
    print("CHRONIC KIDNEY DISEASE CLASSIFICATION")
    print("Enhanced k-Nearest Neighbors with Optimal PCA")
    print("="*80)
    
    # Initialize classifier
    ckd = CKDClassifier(data_path=data_path)
    
    # Step 1: Load data
    ckd.load_data()
    ckd.prepare_target()
    
    # Step 2: EDA
    numeric_cols, categorical_cols = ckd.exploratory_data_analysis()
    ckd.visualize_eda()
    
    # Step 3: Data cleaning
    X, y = ckd.data_cleaning()
    
    # Step 4: Feature engineering
    mi_scores = ckd.feature_engineering()
    
    # Step 5: Train-test split
    ckd.split_data(test_size=0.2)
    
    # Step 6: Build pipeline
    pipeline = ckd.build_pipeline()
    
    # Step 7: Hyperparameter tuning
    best_model, grid_search = ckd.hyperparameter_tuning(pipeline)
    
    # Step 8: PCA analysis
    ckd.analyze_pca()
    
    # Step 9: K-value analysis
    ckd.evaluate_k_values()
    
    # Step 10: Comprehensive evaluation
    eval_results = ckd.comprehensive_evaluation()
    
    # Step 11: Save model
    ckd.save_model()
    
    # Step 12: Generate report
    ckd.generate_report(eval_results, grid_search)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. eda_visualization.png - EDA plots")
    print("  2. feature_importance.png - Feature importance analysis")
    print("  3. pca_analysis.png - PCA component analysis")
    print("  4. k_value_analysis.png - K-value performance")
    print("  5. model_evaluation.png - Comprehensive evaluation plots")
    print("  6. classification_report.txt - Detailed text report")
    print("  7. ckd_summary.json - JSON summary")
    print("  8. knn_ckd_final_model.joblib - Trained model")
    print("  9. ckd_preprocess_info.joblib - Preprocessing info")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()