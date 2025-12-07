# main_clean.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import arff
from sklearn.utils import shuffle

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


def main(data_path='data/chronic_kidney_disease.arff'):
    print("Loading dataset...")
    df = load_arff(data_path)
    df = df.replace('?', np.nan)
    df = normalize_column_strings(df)

    # ensure expected columns lowercased
    df.columns = [c.strip().lower() for c in df.columns]

    df = prepare_target(df)

    print("Dataset shape:", df.shape)
    print("Missing values (top):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # define numeric / categorical columns (from dataset description)
    numeric_candidates = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ['class']]

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
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

    # After imputation, for originally categorical cols round to ints
    for col in categorical_cols:
        if col in X_imputed.columns:
            X_imputed[col] = X_imputed[col].round().astype(int)

    # Update X to imputed version
    X = X_imputed.copy()

    # Scale numeric features (we will use a ColumnTransformer so PCA sees scaled data)
    num_feats = [c for c in numeric_cols if c in X.columns]
    cat_feats = [c for c in categorical_cols if c in X.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
    ], remainder='drop')

    # Split now (we will fit preprocessors on training data only)
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y,
                                                        random_state=RANDOM_STATE)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class distribution:\n", pd.Series(y_train).value_counts())

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

    param_grid = {
        'pca__n_components': n_comp_choices,
        'clf__n_neighbors': [1,3,4,5,7,9,11],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1,2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)

    print("Starting GridSearchCV...")
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV ROC-AUC:", gs.best_score_)

    best_model = gs.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

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

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and preprocessing artefacts
    out_dir = Path('.')
    joblib.dump(best_model, out_dir / 'knn_ckd_final_model.joblib')
    # Save label encoders & feature lists for later use
    joblib.dump({'label_encoders': label_encoders, 'numeric_features': num_feats, 'categorical_features': cat_feats},
                out_dir / 'ckd_preprocess_info.joblib')

    # Optionally save cv results and a small report json
    summary = {
        'best_params': gs.best_params_,
        'cv_best_score': float(gs.best_score_),
        'test_metrics': {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
                         'roc_auc': float(rocauc) if rocauc is not None else None}
    }
    with open(out_dir / 'ckd_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSaved model to 'knn_ckd_final_model.joblib' and summary to 'ckd_summary.json'.")


if __name__ == "__main__":
    main()
