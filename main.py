# main_plots.py
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

from sklearn.impute import KNNImputer
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
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# ---------- Helpers: ARFF loader and normalization ----------
def load_arff(path):
    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        return df
    except Exception:
        # fallback manual parse
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


def normalize_strings(df):
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({'nan': None})
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def prepare_target(df):
    df['class'] = df['class'].astype(str).str.strip().str.lower()
    # fix obvious noise
    df['class'] = df['class'].replace({'no': 'notckd', 'not ckd': 'notckd', 'not_ckd': 'notckd'})
    mapped = df['class'].map({'ckd': 1, 'notckd': 0})
    # drop unmapped rows (rare)
    df = df.loc[mapped.notna()].copy()
    df['class'] = mapped.astype(int)
    return df


# ---------- Plotting helpers ----------
def save_fig(fig, fname):
    out = Path('.')
    fig.savefig(out / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_eda(df, numeric_cols):
    # class distribution
    fig, ax = plt.subplots()
    class_counts = df['class'].map({1:'ckd',0:'notckd'}).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, ax=ax, palette=['#2ecc71','#e74c3c'], legend=False)
    ax.set_title('Class distribution')
    for i,v in enumerate(class_counts.values):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    save_fig(fig, 'eda_class_distribution.png')

    # missing values
    miss = df.isna().sum()
    miss = miss[miss>0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=miss.index, y=miss.values, hue=miss.index, ax=ax, palette='viridis', legend=False)
    ax.set_title('Missing values per feature')
    ax.set_ylabel('Missing count')
    ax.tick_params(axis='x', rotation=90)
    save_fig(fig, 'eda_missing_values.png')

    # correlation heatmap for numeric columns
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
    ax.set_title('Correlation heatmap (numeric features)')
    save_fig(fig, 'eda_correlation_heatmap.png')

    # age distribution per class (if age exists)
    if 'age' in numeric_cols:
        fig, ax = plt.subplots()
        for lab, grp in df.groupby('class'):
            sns.histplot(grp['age'].dropna().astype(float), kde=False, label=('ckd' if lab==1 else 'notckd'), ax=ax, bins=20, alpha=0.6)
        ax.set_title('Age distribution by class')
        ax.legend()
        save_fig(fig, 'eda_age_distribution.png')


def plot_feature_importance(mi_series):
    fig, ax = plt.subplots(figsize=(8,10))
    mi_series.sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature importance (Mutual Information)')
    save_fig(fig, 'feature_importance.png')


def plot_pca(pca, fname_prefix='pca_analysis'):
    var = pca.explained_variance_ratio_
    cumsum = np.cumsum(var)
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    axes[0].bar(range(1, len(var)+1), var, color='steelblue')
    axes[0].set_xlabel('Principal component')
    axes[0].set_ylabel('Explained variance ratio')
    axes[0].set_title('Explained variance per component')

    axes[1].plot(range(1, len(cumsum)+1), cumsum, marker='o')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of components')
    axes[1].set_ylabel('Cumulative explained variance')
    axes[1].legend()
    save_fig(fig, f'{fname_prefix}.png')


def plot_k_vs_accuracy(k_range, train_scores, test_scores, fname='k_value_analysis.png'):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(k_range, train_scores, marker='o', label='Train accuracy')
    ax.plot(k_range, test_scores, marker='s', label='Test accuracy')
    ax.set_xlabel('k (number of neighbors)')
    ax.set_ylabel('Accuracy')
    ax.set_title('k-NN performance vs k')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, fname)


def plot_model_evaluation(y_test, y_pred, y_proba, target_names):
    # confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=target_names, yticklabels=target_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    save_fig(fig, 'model_confusion_matrix.png')

    # ROC curve (binary)
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0,1], [0,1], linestyle='--', color='gray')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
        ax.legend()
        save_fig(fig, 'model_roc_curve.png')

    # metrics bar chart
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    fig, ax = plt.subplots()
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    ax.bar(metrics.keys(), metrics.values(), color=['#3498db','#2ecc71','#e74c3c','#f39c12'])
    ax.set_ylim(0,1.05)
    ax.set_title('Performance metrics')
    for i,(k,v) in enumerate(metrics.items()):
        ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
    save_fig(fig, 'model_metrics_bar.png')


# ---------- Main pipeline ----------
def main(data_path='data/chronic_kidney_disease.arff'):
    print("Loading dataset...")
    df = load_arff(data_path)
    df = df.replace('?', np.nan)
    df = normalize_strings(df)
    df = prepare_target(df)

    print("Dataset shape:", df.shape)
    print("Missing value counts (top):")
    print(df.isna().sum().sort_values(ascending=False).head(30))

    # identify numeric and categorical
    numeric_candidates = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ['class']]

    # EDA plots
    plot_eda(df, numeric_cols)

    # Coerce numeric types
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # optional derived feature
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'].astype(float), bins=[0,18,35,50,65,200],
                                 labels=['child','young','adult','mid','senior'])
        if 'age_group' not in categorical_cols:
            categorical_cols.append('age_group')

    # Separate X,y
    X = df.drop(columns=['class']).reset_index(drop=True)
    y = df['class'].reset_index(drop=True)

    # Label encode categorical for KNN imputer
    label_encoders = {}
    X_enc = X.copy()
    for col in categorical_cols:
        if col not in X_enc.columns:
            continue
        # Convert to object type first to avoid categorical issues
        X_enc[col] = X_enc[col].astype(object)
        le = LabelEncoder()
        mask = X_enc[col].notna()
        if mask.sum() > 0:
            X_enc.loc[mask, col] = le.fit_transform(X_enc.loc[mask, col])
            label_encoders[col] = le
        else:
            X_enc[col] = np.nan
        X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce')

    # KNN impute
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

    # Round categorical back
    for col in categorical_cols:
        if col in X_imputed.columns:
            X_imputed[col] = X_imputed[col].round().astype(int)

    # Feature importance (mutual info)
    mi = mutual_info_classif(X_imputed, y, discrete_features='auto', random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=X_imputed.columns).sort_values(ascending=False)
    plot_feature_importance(mi_series.head(30))

    # Train/test split (shuffle)
    X_all = X_imputed.copy()
    X_all, y = shuffle(X_all, y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.20,
                                                        stratify=y, random_state=RANDOM_STATE)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Class distribution (train):")
    print(pd.Series(y_train).value_counts())

    # Build preprocessing transformer: scale numeric, one-hot categorical (so PCA receives numeric only)
    num_feats = [c for c in numeric_cols if c in X_all.columns]
    cat_feats = [c for c in categorical_cols if c in X_all.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
    ], remainder='drop')

    # Quick check: encoded dimension
    encoded_dim = preprocessor.fit_transform(X_train.iloc[:5]).shape[1]
    print("Estimated encoded dim (sample):", encoded_dim)

    # Full pipeline with PCA and KNN (PCA n_components tuned)
    pipe = Pipeline([
        ('pre', preprocessor),
        ('pca', PCA(random_state=RANDOM_STATE)),
        ('clf', KNeighborsClassifier())
    ])

    # Param grid (tunable)
    n_comp_choices = sorted(list({min(5, encoded_dim), min(8, encoded_dim), min(10, encoded_dim),
                                  min(12, encoded_dim), min(15, encoded_dim), encoded_dim}))
    n_comp_choices = [int(c) for c in n_comp_choices if c >= 1]

    param_grid = {
        'pca__n_components': n_comp_choices,
        'clf__n_neighbors': [1,3,4,5,7,9,11,15],
        'clf__weights': ['uniform','distance'],
        'clf__p': [1,2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    print("Starting GridSearchCV...")
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    print("Best params:", gs.best_params_)
    print("Best CV score (ROC-AUC):", gs.best_score_)

    # Evaluate on test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("\nTest metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    if rocauc is not None:
        print(f"ROC-AUC: {rocauc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['notckd','ckd']))

    # Generate evaluation plots
    plot_model_evaluation(y_test, y_pred, y_proba, target_names=['notckd','ckd'])

    # ---------- k vs accuracy sweep (on original scaled+encoded data)
    # We'll build a pipeline without PCA and compute train/test accuracy across k
    pre_encoded = preprocessor.fit(X_train)
    X_train_enc = pre_encoded.transform(X_train)
    X_test_enc = pre_encoded.transform(X_test)

    train_scores = []
    test_scores = []
    k_range = list(range(1, 31))
    print("Evaluating k sweep (1..30)...")
    for k in k_range:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_enc, y_train)
        train_scores.append(accuracy_score(y_train, knn_k.predict(X_train_enc)))
        test_scores.append(accuracy_score(y_test, knn_k.predict(X_test_enc)))

    plot_k_vs_accuracy(k_range, train_scores, test_scores)

    # ---------- PCA comparison: evaluate best k with and without PCA (used earlier style)
    # Get best k from k sweep (on test_scores)
    best_k_index = int(np.argmax(test_scores))
    best_k = k_range[best_k_index]
    print("Best k from sweep (no PCA):", best_k, "with test accuracy:", test_scores[best_k_index])

    # Without PCA evaluation (retrain knn with best_k)
    knn_no_pca = KNeighborsClassifier(n_neighbors=best_k)
    knn_no_pca.fit(X_train_enc, y_train)
    no_pca_test_acc = accuracy_score(y_test, knn_no_pca.predict(X_test_enc))

    # With PCA: use same preprocessing + PCA with components chosen in best_model
    pca_comp = best_model.named_steps['pca'].n_components
    pca = PCA(n_components=pca_comp, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_enc)
    X_test_pca = pca.transform(X_test_enc)
    knn_pca = KNeighborsClassifier(n_neighbors=best_k)
    knn_pca.fit(X_train_pca, y_train)
    with_pca_test_acc = accuracy_score(y_test, knn_pca.predict(X_test_pca))

    # Save PCA plots
    plot_pca(best_model.named_steps['pca'], fname_prefix='pca_analysis')

    # PCA comparison plot
    fig, ax = plt.subplots(figsize=(8,5))
    categories = ['No PCA', 'With PCA']
    accuracies = [no_pca_test_acc, with_pca_test_acc]
    features = [X_train_enc.shape[1], pca_comp]
    ax.bar(categories, accuracies, color=['steelblue','coral'])
    ax.set_ylim([0,1])
    ax.set_title('Accuracy: No PCA vs With PCA')
    for i,v in enumerate(accuracies):
        ax.text(i, v+0.02, f'{v*100:.2f}%', ha='center')
    save_fig(fig, 'pca_accuracy_comparison.png')

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(categories, features, color=['steelblue','coral'])
    ax.set_title('Feature count: No PCA vs With PCA')
    for i,v in enumerate(features):
        ax.text(i, v+0.5, str(v), ha='center')
    save_fig(fig, 'pca_featurecount_comparison.png')

    # Save final model and preprocess info
    joblib.dump(best_model, Path('.') / 'knn_ckd_final_model.joblib')
    joblib.dump({'label_encoders': label_encoders, 'numeric_features': num_feats, 'categorical_features': cat_feats, 'preprocessor': preprocessor},
                Path('.') / 'ckd_preprocess_info.joblib')

    summary = {
        'best_params': gs.best_params_,
        'cv_best_score': float(gs.best_score_),
        'test_metrics': {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
                        'roc_auc': float(rocauc) if rocauc is not None else None},
        'pca_components_used': int(best_model.named_steps['pca'].n_components)
    }
    with open(Path('.') / 'ckd_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Saved plots and model artifacts. Done.")


if __name__ == '__main__':
    main('data/chronic_kidney_disease.arff')
