"""
Chronic Kidney Disease Classification using k-Nearest Neighbors
CSCI 31022 - Machine Learning and Pattern Recognition
Assignment 1

This script performs comprehensive analysis and classification of the 
Chronic Kidney Disease dataset using k-NN classifier with various 
preprocessing techniques and hyperparameter tuning.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.io import arff
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve, auc)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_arff(path):
    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        # decode byte strings if present
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        return df
    except Exception:
        # robust manual parser fallback
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


def normalize_column_strings(df):
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({'nan': None})
    return df


def prepare_target(df):
    # normalize target and map
    df['class'] = df['class'].astype(str).str.strip().str.lower()
    # fix obvious variants, then map; drop unmapped
    df['class'] = df['class'].replace({'no': 'notckd', 'not ckd': 'notckd', 'not_ckd': 'notckd'})
    mapped = df['class'].map({'ckd': 1, 'notckd': 0})
    n_unmapped = mapped.isna().sum()
    if n_unmapped > 0:
        df = df.loc[mapped.notna()].copy()
        mapped = mapped.loc[mapped.notna()]
    df['class'] = mapped.astype(int)
    return df


def visualize_eda(df, numeric_cols, categorical_cols):
    """Create comprehensive EDA visualizations."""
    print("\n--- Creating EDA Visualizations ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Class Distribution
    class_counts = df['class'].value_counts()
    axes[0, 0].bar(['notckd', 'ckd'], [class_counts.get(0, 0), class_counts.get(1, 0)], 
                   color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate([class_counts.get(0, 0), class_counts.get(1, 0)]):
        axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # 2. Missing values
    missing_data = df.isnull().sum().sort_values(ascending=False)[:15]
    if missing_data.sum() > 0:
        axes[0, 1].barh(range(len(missing_data)), missing_data.values, color='coral')
        axes[0, 1].set_yticks(range(len(missing_data)))
        axes[0, 1].set_yticklabels(missing_data.index, fontsize=9)
        axes[0, 1].set_title('Top 15 Features with Missing Values', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Missing Count')
        axes[0, 1].invert_yaxis()
    
    # 3. Correlation heatmap (numeric features only)
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    correlation = numeric_df.corr()
    im = axes[1, 0].imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_title('Correlation Heatmap (Numeric Features)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(correlation.columns)))
    axes[1, 0].set_yticks(range(len(correlation.columns)))
    axes[1, 0].set_xticklabels(correlation.columns, rotation=90, fontsize=8)
    axes[1, 0].set_yticklabels(correlation.columns, fontsize=8)
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Age distribution by class
    if 'age' in numeric_cols:
        for class_label in [0, 1]:
            subset = df[df['class'] == class_label]['age'].dropna()
            axes[1, 1].hist(pd.to_numeric(subset, errors='coerce'), alpha=0.6, 
                          label=['notckd', 'ckd'][class_label], bins=20)
        axes[1, 1].set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ EDA visualization saved as 'eda_visualization.png'")
    plt.close()


def visualize_feature_importance(X, y, feature_names):
    """Visualize feature importance using mutual information."""
    print("\n--- Calculating and Visualizing Feature Importance ---")
    
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_scores = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(mi_scores.head(10))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    mi_scores.sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Importance (Mutual Information)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mutual Information Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved as 'feature_importance.png'")
    plt.close()
    
    return mi_scores


def visualize_pca_analysis(pca, variance_threshold=0.95):
    """Create PCA analysis visualizations."""
    print("\n--- Creating PCA Analysis Visualizations ---")
    
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
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, marker='o', linestyle='-', 
                color='darkblue', linewidth=2)
    axes[1].axhline(y=variance_threshold, color='r', linestyle='--', 
                   label=f'{variance_threshold*100}% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ PCA analysis plot saved as 'pca_analysis.png'")
    plt.close()


def evaluate_k_values(X_train, X_test, y_train, y_test, k_range=range(1, 21)):
    """Evaluate different k values and create visualization."""
    print("\n--- Evaluating Different K Values ---")
    
    train_scores = []
    test_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)
        
        train_scores.append(accuracy_score(y_train, train_pred))
        test_scores.append(accuracy_score(y_test, test_pred))
    
    best_k = k_range[np.argmax(test_scores)]
    best_score = max(test_scores)
    
    print(f"Best k value: {best_k} with accuracy: {best_score*100:.2f}%")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k_range, train_scores, label='Training Accuracy', marker='o', 
           linestyle='-', linewidth=2, color='blue')
    ax.plot(k_range, test_scores, label='Testing Accuracy', marker='s', 
           linestyle='-', linewidth=2, color='orange')
    ax.axvline(x=best_k, color='r', linestyle='--', linewidth=2, 
              label=f'Best k={best_k}')
    ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('k-NN Performance vs. k Value', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('k_value_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ K-value analysis plot saved as 'k_value_analysis.png'")
    plt.close()
    
    return best_k, best_score


def visualize_model_evaluation(y_test, y_pred, y_proba, target_names=['notckd', 'ckd']):
    """Create comprehensive model evaluation visualizations."""
    print("\n--- Creating Model Evaluation Visualizations ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=target_names, yticklabels=target_names, 
                cbar_kws={'label': 'Count'})
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction Distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    true_counts = pd.Series(y_test).value_counts().sort_index()
    
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
    
    # 4. Performance Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    axes[1, 1].bar(metrics.keys(), metrics.values(), color=colors)
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    for i, (k, v) in enumerate(metrics.items()):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("✓ Model evaluation plots saved as 'model_evaluation.png'")
    plt.close()


def main(data_path='data/chronic_kidney_disease.arff'):
    print("="*80)
    print("CHRONIC KIDNEY DISEASE CLASSIFICATION USING k-NN")
    print("="*80)
    
    print("\n[STEP 1] Loading dataset...")
    df = load_arff(data_path)
    df = df.replace('?', np.nan)
    df = normalize_column_strings(df)

    # ensure expected columns lowercased
    df.columns = [c.strip().lower() for c in df.columns]

    df = prepare_target(df)

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values (top 20):")
    missing_summary = df.isna().sum().sort_values(ascending=False).head(20)
    print(missing_summary)

    # define numeric / categorical columns (from dataset description)
    numeric_candidates = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ['class']]
    
    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # EDA Visualizations
    print("\n[STEP 2] Exploratory Data Analysis...")
    visualize_eda(df, numeric_cols, categorical_cols)

    # coerce numeric columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # optionally create a simple derived feature (age group) - keep as categorical
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'].astype(float), bins=[0,18,35,50,65,200],
                                 labels=['child','young','adult','mid','senior'])
        if 'age_group' not in categorical_cols:
            categorical_cols.append('age_group')

    # Split X,y early for imputation/encoding
    X = df.drop(columns=['class']).reset_index(drop=True)
    y = df['class'].reset_index(drop=True)

    print("\n[STEP 3] Data Preprocessing...")
    # Label-encode categorical columns (needed for KNN imputer)
    label_encoders = {}
    X_enc = X.copy()
    
    # Convert categorical columns to object type to avoid categorical assignment issues
    for col in categorical_cols:
        if col in X_enc.columns and X_enc[col].dtype.name == 'category':
            X_enc[col] = X_enc[col].astype(object)
    
    for col in categorical_cols:
        if col not in X_enc.columns:
            continue
        le = LabelEncoder()
        mask = X_enc[col].notna()
        if mask.sum() > 0:
            X_enc.loc[mask, col] = le.fit_transform(X_enc.loc[mask, col])
            label_encoders[col] = le
        else:
            X_enc[col] = np.nan
        X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce')

    # KNN Imputation for all features
    print("Applying KNN Imputation (k=5)...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

    # After imputation, for originally categorical cols round to ints
    for col in categorical_cols:
        if col in X_imputed.columns:
            X_imputed[col] = X_imputed[col].round().astype(int)

    # Update X to imputed version
    X = X_imputed.copy()
    feature_names = X.columns.tolist()

    # Feature Importance Analysis
    print("\n[STEP 4] Feature Importance Analysis...")
    mi_scores = visualize_feature_importance(X, y, feature_names)

    # Scale numeric features (we will use a ColumnTransformer so PCA sees scaled data)
    num_feats = [c for c in numeric_cols if c in X.columns]
    cat_feats = [c for c in categorical_cols if c in X.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
    ], remainder='drop')

    # Split now (we will fit preprocessors on training data only)
    print("\n[STEP 5] Train-Test Split (80-20)...")
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=RANDOM_STATE)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class distribution:\n{pd.Series(y_train).value_counts()}")

    # Build pipeline: preprocessor -> PCA (tunable) -> KNN (tunable)
    pipe = Pipeline([
        ('pre', preprocessor),
        ('pca', PCA(random_state=RANDOM_STATE)),
        ('clf', KNeighborsClassifier())
    ])

    # Estimate encoded feature count to choose PCA options
    sample_encoded = preprocessor.fit_transform(X_train.iloc[:10])
    encoded_dim = sample_encoded.shape[1]
    n_comp_choices = sorted(list({min(5, encoded_dim), min(8, encoded_dim), min(10, encoded_dim),
                                  min(12, encoded_dim), min(15, encoded_dim), encoded_dim}))
    # remove duplicates and keep valid
    n_comp_choices = [int(c) for c in n_comp_choices if c >= 1]

    print("\n[STEP 6] K-Value Analysis...")
    # First, do a quick k-value analysis with preprocessed data
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    best_k_initial, best_score_initial = evaluate_k_values(X_train_pre, X_test_pre, 
                                                           y_train, y_test, k_range=range(1, 21))

    print("\n[STEP 7] PCA Dimensionality Reduction...")
    # Apply PCA for visualization
    pca_viz = PCA(n_components=min(12, encoded_dim), random_state=RANDOM_STATE)
    X_train_pca_viz = pca_viz.fit_transform(X_train_pre)
    visualize_pca_analysis(pca_viz, variance_threshold=0.95)
    
    explained_var = np.sum(pca_viz.explained_variance_ratio_)
    print(f"PCA with {pca_viz.n_components} components explains {explained_var*100:.2f}% variance")

    print("\n[STEP 8] Hyperparameter Tuning with GridSearchCV...")
    param_grid = {
        'pca__n_components': n_comp_choices,
        'clf__n_neighbors': [1,3,5,7,9,11,13,15],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1,2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)

    print("Starting GridSearchCV (this may take a few minutes)...")
    gs.fit(X_train, y_train)

    print(f"\nBest params: {gs.best_params_}")
    print(f"Best CV ROC-AUC: {gs.best_score_:.4f}")

    best_model = gs.best_estimator_

    # Evaluate on test set
    print("\n[STEP 9] Model Evaluation...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("\nTest metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if rocauc is not None:
        print(f"ROC-AUC:   {rocauc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['notckd','ckd']))

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualize model evaluation
    visualize_model_evaluation(y_test, y_pred, y_proba, target_names=['notckd', 'ckd'])

    # Cross-validation analysis
    print("\n[STEP 10] Cross-Validation Analysis...")
    cv_scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

    # Save model and preprocessing artefacts
    print("\n[STEP 11] Saving Model and Results...")
    out_dir = Path('.')
    joblib.dump(best_model, out_dir / 'knn_ckd_final_model.joblib')
    # Save label encoders & feature lists for later use
    joblib.dump({'label_encoders': label_encoders, 'numeric_features': num_feats, 
                 'categorical_features': cat_feats},
                out_dir / 'ckd_preprocess_info.joblib')

    # Optionally save cv results and a small report json
    summary = {
        'best_params': gs.best_params_,
        'cv_best_score': float(gs.best_score_),
        'test_metrics': {
            'accuracy': float(acc), 
            'precision': float(prec), 
            'recall': float(rec), 
            'f1': float(f1),
            'roc_auc': float(rocauc) if rocauc is not None else None
        },
        'optimal_k': int(gs.best_params_['clf__n_neighbors']),
        'pca_components': int(gs.best_params_['pca__n_components']),
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    }
    with open(out_dir / 'ckd_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Model saved to 'knn_ckd_final_model.joblib'")
    print("✓ Summary saved to 'ckd_summary.json'")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. eda_visualization.png - EDA plots")
    print("  2. feature_importance.png - Feature importance analysis")
    print("  3. pca_analysis.png - PCA variance analysis")
    print("  4. k_value_analysis.png - K-value optimization")
    print("  5. model_evaluation.png - Comprehensive evaluation plots")
    print("  6. knn_ckd_final_model.joblib - Trained model")
    print("  7. ckd_summary.json - Performance summary")
    print("  8. ckd_preprocess_info.joblib - Preprocessing info")
    print("\n" + "="*80)
    
    print(f"\n🎯 OPTIMAL K VALUE: {gs.best_params_['clf__n_neighbors']}")
    print(f"📊 FINAL TEST ACCURACY: {acc*100:.2f}%")
    print(f"🎨 PCA COMPONENTS: {gs.best_params_['pca__n_components']}")
    print("="*80)


if __name__ == "__main__":
    main()
