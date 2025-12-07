"""
Chronic Kidney Disease Classification using k-Nearest Neighbors
CSCI 31022 - Machine Learning and Pattern Recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class CKDClassifier:
    def __init__(self, data_path='data/chronic_kidney_disease.arff'):
        self.data_path = data_path
        self.df = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.label_encoders = {}
        self.scaler, self.pca, self.best_model = None, None, None
        self.feature_names = None
        
    def _manual_arff_parse(self, filepath):
        attributes, data_rows, data_section = [], [], False
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'): continue
                if line.lower() == '@data': data_section = True; continue
                if line.lower().startswith('@attribute'):
                    attributes.append(line.split()[1].strip("'\""))
                elif data_section:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(attributes): data_rows.append(values)
        return pd.DataFrame(data_rows, columns=attributes).replace('?', np.nan)
    
    def load_data(self):
        print("="*80 + "\nSTEP 1: LOADING DATA\n" + "="*80)
        try:
            data, meta = arff.loadarff(self.data_path)
            self.df = pd.DataFrame(data)
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    self.df[col] = self.df[col].str.decode('utf-8')
        except:
            self.df = self._manual_arff_parse(self.data_path)
        print(f"\nDataset: {self.df.shape[0]} samples, {self.df.shape[1]-1} features")
        return self.df
    
    def exploratory_data_analysis(self):
        print("\n" + "="*80 + "\nSTEP 2: EXPLORATORY DATA ANALYSIS\n" + "="*80)
        print("\n--- Dataset Info ---\n", self.df.info())
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        print(pd.DataFrame({'Count': missing[missing > 0], 'Percentage': (missing[missing > 0]/len(self.df)*100)}))
        print("\n--- Class Distribution ---\n", self.df['class'].value_counts())
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in self.df.select_dtypes(include=['object']).columns if c != 'class']
        print(f"\nNumeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
        return numeric_cols, categorical_cols
    
    def visualize_eda(self, numeric_cols, categorical_cols):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        class_counts = self.df['class'].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Class Distribution', fontweight='bold')
        for i, v in enumerate(class_counts.values): axes[0, 0].text(i, v+5, str(v), ha='center', fontweight='bold')
        
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            axes[0, 1].bar(range(len(missing_data)), missing_data.values)
            axes[0, 1].set_title('Missing Values per Feature', fontweight='bold')
        
        correlation = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce').corr()
        im = axes[1, 0].imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Correlation Heatmap', fontweight='bold')
        axes[1, 0].set_xticks(range(len(correlation.columns)))
        axes[1, 0].set_xticklabels(correlation.columns, rotation=90, fontsize=8)
        plt.colorbar(im, ax=axes[1, 0])
        
        if 'age' in numeric_cols:
            for cls in self.df['class'].unique():
                axes[1, 1].hist(self.df[self.df['class']==cls]['age'].dropna(), alpha=0.6, label=cls, bins=20)
            axes[1, 1].set_title('Age Distribution by Class', fontweight='bold')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        print("\n--- Saved: eda_visualization.png")
        plt.close()
        
    def data_cleaning(self, numeric_cols, categorical_cols):
        print("\n" + "="*80 + "\nSTEP 3: DATA CLEANING AND PREPROCESSING\n" + "="*80)
        df_clean = self.df.replace('?', np.nan)
        for col in numeric_cols: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        X, y = df_clean.drop('class', axis=1), df_clean['class']
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col in X_encoded.columns:
                le = LabelEncoder()
                mask = X_encoded[col].notna()
                if mask.sum() > 0:
                    X_encoded.loc[mask, col] = le.fit_transform(X_encoded.loc[mask, col])
                    self.label_encoders[col] = le
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
        
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)
        for col in categorical_cols:
            if col in X_imputed.columns: X_imputed[col] = X_imputed[col].round().astype(int)
        
        le_target = LabelEncoder()
        self.X, self.y = X_imputed, le_target.fit_transform(y)
        self.label_encoders['class'] = le_target
        self.feature_names = X_imputed.columns.tolist()
        print(f"\nFinal shape: {self.X.shape}, Missing: {self.X.isnull().sum().sum()}")
        return self.X, self.y
    
    def feature_engineering(self):
        print("\n" + "="*80 + "\nSTEP 4: FEATURE IMPORTANCE\n" + "="*80)
        mi_scores = pd.Series(mutual_info_classif(self.X, self.y, random_state=RANDOM_STATE), 
                             index=self.feature_names).sort_values(ascending=False)
        print("\nTop 10 Features:\n", mi_scores.head(10))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        mi_scores.sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Feature Importance', fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n--- Saved: feature_importance.png")
        plt.close()
        return mi_scores
    
    def feature_scaling(self):
        print("\n" + "="*80 + "\nSTEP 5: FEATURE SCALING\n" + "="*80)
        self.scaler = StandardScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.feature_names)
        print("Applied StandardScaler")
        return self.X
    
    def split_data(self, test_size=0.2):
        print("\n" + "="*80 + "\nSTEP 6: TRAIN-TEST SPLIT\n" + "="*80)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=RANDOM_STATE, stratify=self.y)
        print(f"Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def dimensionality_reduction(self, n_components=None, variance_threshold=0.95):
        print("\n" + "="*80 + "\nSTEP 7: PCA\n" + "="*80)
        if n_components is None:
            pca_temp = PCA(random_state=RANDOM_STATE).fit(self.X_train)
            n_components = np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = self.pca.fit_transform(self.X_train)
        X_test_pca = self.pca.transform(self.X_test)
        print(f"Reduced: {self.X_train.shape[1]} → {n_components} ({np.sum(self.pca.explained_variance_ratio_)*100:.2f}% variance)")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_)+1), self.pca.explained_variance_ratio_, color='steelblue')
        axes[0].set_title('Explained Variance by Component', fontweight='bold')
        
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum)+1), cumsum, marker='o', color='darkblue')
        axes[1].axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100}%')
        axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        print("--- Saved: pca_analysis.png")
        plt.close()
        return X_train_pca, X_test_pca
    
    def evaluate_k_values(self, X_train, X_test, k_range=range(1, 31)):
        print("\n" + "="*80 + "\nSTEP 8: K-VALUE ANALYSIS\n" + "="*80)
        train_scores, test_scores = [], []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, self.y_train)
            train_scores.append(accuracy_score(self.y_train, knn.predict(X_train)))
            test_scores.append(accuracy_score(self.y_test, knn.predict(X_test)))
        
        best_k = k_range[np.argmax(test_scores)]
        print(f"Best k: {best_k}, Accuracy: {max(test_scores)*100:.2f}%")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(k_range, train_scores, label='Train', marker='o', linewidth=2)
        ax.plot(k_range, test_scores, label='Test', marker='s', linewidth=2)
        ax.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
        ax.set_title('k-NN Performance vs k Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('k_value_analysis.png', dpi=300, bbox_inches='tight')
        print("--- Saved: k_value_analysis.png")
        plt.close()
        return best_k, max(test_scores), train_scores, test_scores
    
    def hyperparameter_tuning(self, X_train, X_test):
        print("\n" + "="*80 + "\nSTEP 9: HYPERPARAMETER TUNING\n" + "="*80)
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
        
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        
        print(f"\nBest params: {grid_search.best_params_}")
        print(f"CV accuracy: {grid_search.best_score_*100:.2f}%")
        print(f"Test accuracy: {accuracy_score(self.y_test, self.best_model.predict(X_test))*100:.2f}%")
        return self.best_model, grid_search
    
    def comprehensive_evaluation(self, X_test):
        print("\n" + "="*80 + "\nSTEP 10: MODEL EVALUATION\n" + "="*80)
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, average='weighted')
        rec = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\nAccuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
        print("\n", classification_report(self.y_test, y_pred, target_names=self.label_encoders['class'].classes_))
        
        if len(np.unique(self.y_test)) == 2:
            print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba[:, 1]):.4f}")
        
        self._visualize_evaluation(y_pred, y_pred_proba)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
    
    def _visualize_evaluation(self, y_pred, y_pred_proba):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        target_names = self.label_encoders['class'].classes_
        
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                   xticklabels=target_names, yticklabels=target_names)
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        
        if len(np.unique(self.y_test)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={auc(fpr, tpr):.2f}')
            axes[0, 1].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
            axes[0, 1].set_title('ROC Curve', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        x = np.arange(len(target_names))
        axes[1, 0].bar(x-0.2, pd.Series(self.y_test).value_counts().sort_index().values, 0.4, label='True', color='steelblue')
        axes[1, 0].bar(x+0.2, pd.Series(y_pred).value_counts().sort_index().values, 0.4, label='Predicted', color='coral')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(target_names)
        axes[1, 0].set_title('True vs Predicted', fontweight='bold')
        axes[1, 0].legend()
        
        metrics = {'Accuracy': accuracy_score(self.y_test, y_pred), 'Precision': precision_score(self.y_test, y_pred, average='weighted'),
                  'Recall': recall_score(self.y_test, y_pred, average='weighted'), 'F1': f1_score(self.y_test, y_pred, average='weighted')}
        axes[1, 1].bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_title('Performance Metrics', fontweight='bold')
        for i, (k, v) in enumerate(metrics.items()): axes[1, 1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("--- Saved: model_evaluation.png")
        plt.close()
    
    def compare_with_without_pca(self):
        print("\n" + "="*80 + "\nSTEP 11: PCA COMPARISON\n" + "="*80)
        results = {}
        
        # No PCA
        best_k, best_score, _, _ = self.evaluate_k_values(self.X_train, self.X_test, range(1, 21))
        knn = KNeighborsClassifier(n_neighbors=best_k).fit(self.X_train, self.y_train)
        results['no_pca'] = {'accuracy': accuracy_score(self.y_test, knn.predict(self.X_test)),
                            'best_k': best_k, 'n_features': self.X_train.shape[1]}
        
        # Multiple PCA configs
        configs = [{'name': 'PCA_12', 'n_components': 12}, {'name': 'PCA_14', 'n_components': 14},
                  {'name': 'PCA_16', 'n_components': 16}, {'name': 'PCA_95pct', 'variance_threshold': 0.95}]
        
        pca_results, best_pca = [], None
        for cfg in configs:
            if 'n_components' in cfg:
                pca = PCA(n_components=cfg['n_components'], random_state=RANDOM_STATE)
            else:
                n = np.argmax(np.cumsum(PCA(random_state=RANDOM_STATE).fit(self.X_train).explained_variance_ratio_) >= cfg['variance_threshold']) + 1
                pca = PCA(n_components=n, random_state=RANDOM_STATE)
            
            X_tr, X_te = pca.fit_transform(self.X_train), pca.transform(self.X_test)
            k, _, _, _ = self.evaluate_k_values(X_tr, X_te, range(1, 21))
            knn = KNeighborsClassifier(n_neighbors=k).fit(X_tr, self.y_train)
            acc = accuracy_score(self.y_test, knn.predict(X_te))
            
            # Create label based on config type
            if 'n_components' in cfg:
                label = str(cfg['n_components'])
            else:
                label = f"{cfg['variance_threshold']*100}% Var"
            
            res = {'name': cfg['name'], 'label': label,
                  'accuracy': acc, 'best_k': k, 'n_features': pca.n_components_,
                  'variance': np.sum(pca.explained_variance_ratio_), 'X_train': X_tr, 'X_test': X_te, 'pca': pca}
            pca_results.append(res)
            results[cfg['name']] = res
            if best_pca is None or acc > best_pca['accuracy']: best_pca = res
        
        print(f"\n{'Config':<15} {'Features':<10} {'Variance':<12} {'Accuracy':<12}")
        print("-"*50)
        print(f"{'No PCA':<15} {results['no_pca']['n_features']:<10} {'N/A':<12} {results['no_pca']['accuracy']*100:<11.2f}%")
        for r in pca_results: print(f"{str(r['label']):<15} {r['n_features']:<10} {r['variance']*100:<11.2f}% {r['accuracy']*100:<11.2f}%")
        print(f"\nBest: {best_pca['label']} → {best_pca['accuracy']*100:.2f}%")
        
        self._visualize_pca_comparison(results, pca_results)
        self.pca = best_pca['pca']
        return results, best_pca['X_train'], best_pca['X_test']
    
    def _visualize_pca_comparison(self, results, pca_results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        labels = ['No PCA'] + [r['label'] for r in pca_results]
        accs = [results['no_pca']['accuracy']] + [r['accuracy'] for r in pca_results]
        feats = [results['no_pca']['n_features']] + [r['n_features'] for r in pca_results]
        colors = ['steelblue'] + ['coral', 'lightcoral', 'salmon', 'orangered']
        
        bars = axes[0, 0].bar(range(len(labels)), accs, color=colors)
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=15, ha='right')
        axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
        for bar, acc in zip(bars, accs): axes[0, 0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{acc*100:.2f}%', ha='center', fontweight='bold')
        
        bars = axes[0, 1].bar(range(len(labels)), feats, color=colors)
        axes[0, 1].set_xticks(range(len(labels)))
        axes[0, 1].set_xticklabels(labels, rotation=15, ha='right')
        axes[0, 1].set_title('Feature Count', fontweight='bold')
        for bar, f in zip(bars, feats): axes[0, 1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, str(f), ha='center', fontweight='bold')
        
        reductions = [0] + [(1-r['n_features']/results['no_pca']['n_features'])*100 for r in pca_results]
        axes[1, 0].plot(reductions, [a*100 for a in accs], marker='o', markersize=10, linewidth=2)
        axes[1, 0].set_title('Accuracy vs Feature Reduction', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        var_data = sorted([(r['n_features'], r['variance']*100) for r in pca_results])
        axes[1, 1].plot([v[0] for v in var_data], [v[1] for v in var_data], marker='s', markersize=10, linewidth=2)
        axes[1, 1].axhline(y=95, color='orange', linestyle='--', label='95%')
        axes[1, 1].set_title('Variance by Components', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_comparison.png', dpi=300, bbox_inches='tight')
        print("--- Saved: pca_comparison.png")
        plt.close()


def main():
    print("="*80 + "\nCHRONIC KIDNEY DISEASE CLASSIFICATION\n" + "="*80)
    
    ckd = CKDClassifier(data_path='data/chronic_kidney_disease.arff')
    ckd.load_data()
    numeric_cols, categorical_cols = ckd.exploratory_data_analysis()
    ckd.visualize_eda(numeric_cols, categorical_cols)
    ckd.data_cleaning(numeric_cols, categorical_cols)
    ckd.feature_engineering()
    ckd.feature_scaling()
    ckd.split_data(test_size=0.2)
    pca_results, X_train_pca, X_test_pca = ckd.compare_with_without_pca()
    ckd.hyperparameter_tuning(X_train_pca, X_test_pca)
    ckd.comprehensive_evaluation(X_test_pca)
    
    print("\n" + "="*80 + "\nANALYSIS COMPLETE!\n" + "="*80)
    print("\nGenerated: eda_visualization.png, feature_importance.png, pca_analysis.png,")
    print("           k_value_analysis.png, model_evaluation.png, pca_comparison.png")


if __name__ == "__main__":
    main()