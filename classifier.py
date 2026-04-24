import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Ultimate sledgehammers for Apple Silicon OpenMP Segmentation Faults
os.environ['KMP_DUPLICATE_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

class MetaClassifier:
    def __init__(self, use_logistic_regression=True, xgb_params=None):
        """
        Implements a Stacking Ensemble:
        XGBoost is the primary tabular classifier.
        Logistic Regression is optionally used as a fast, highly-regularized baseline.
        """
        default_params = {
            'n_estimators': 100, 
            'max_depth': 3, 
            'learning_rate': 0.1, 
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': 1,  # CRITICAL: Prevents segmentation fault on macOS M-series chips
            'tree_method': 'hist' # CRITICAL: Bypasses the exact greedy algorithm which triggers the segfault
        }
        
        if xgb_params is not None:
            default_params.update(xgb_params)
            
        self.xgb_model = xgb.XGBClassifier(**default_params)
        
        self.use_lr = use_logistic_regression
        if self.use_lr:
            self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
            
        self.is_trained = False
        
    def concatenate_features(self, cnn_feat, codebert_feat, heuristic_feat=None):
        """
        Flattens and concatenates features from the deep learning models and optional heuristics.
        Inputs are expected to be numpy arrays or torch tensors that can be converted.
        Returns: 1D numpy array of concatenated features.
        """
        def to_flat_numpy(x):
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            return np.array(x).flatten()
            
        cnn_flat = to_flat_numpy(cnn_feat)
        cb_flat = to_flat_numpy(codebert_feat)
        
        if heuristic_feat is not None:
            heuristic_flat = to_flat_numpy(heuristic_feat)
            # Concatenate into one 1D array (128 + 768 + 1 = 897 dimensions)
            concat_vector = np.concatenate([cnn_flat, cb_flat, heuristic_flat])
        else:
            # Concatenate into one 1D array (128 + 768 = 896 dimensions)
            concat_vector = np.concatenate([cnn_flat, cb_flat])
            
        return concat_vector
        
    def train(self, X, y):
        """
        X: List or 2D array of concatenated features
        y: List or 1D array of labels (0 = benign, 1 = phishing)
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"[MetaClassifier] Training XGBoost on {X.shape[0]} samples with {X.shape[1]} features...")
        self.xgb_model.fit(X, y)
        
        if self.use_lr:
            print(f"[MetaClassifier] Training Logistic Regression baseline...")
            self.lr_model.fit(X, y)
            
        self.is_trained = True
        
    def predict_proba(self, feature_vector):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        
        X = np.array(feature_vector).reshape(1, -1)
        xgb_prob = self.xgb_model.predict_proba(X)[0][1]
        
        if self.use_lr:
            lr_prob = self.lr_model.predict_proba(X)[0][1]
            # Average the probabilities for ensemble robustness
            final_prob = (xgb_prob + lr_prob) / 2.0
            return final_prob
        
        return xgb_prob
        
    def predict(self, feature_vector, threshold=0.5):
        prob = self.predict_proba(feature_vector)
        return 1 if prob >= threshold else 0
        
    def save(self, path="meta_classifier.pkl"):
        state = {
            'xgb_model': self.xgb_model,
            'lr_model': self.lr_model if self.use_lr else None,
            'use_lr': self.use_lr
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, path="meta_classifier.pkl"):
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.xgb_model = state['xgb_model']
            self.lr_model = state['lr_model']
            self.use_lr = state['use_lr']
        self.is_trained = True

if __name__ == "__main__":
    clf = MetaClassifier()
    mock_cnn = np.random.rand(1, 128)
    mock_cb  = np.random.rand(1, 768)
    
    vec1 = clf.concatenate_features(mock_cnn, mock_cb)
    vec2 = clf.concatenate_features(mock_cnn, mock_cb)
    print(f"Concatenated feature vector shape: {vec1.shape}")
    
    clf.train([vec1, vec2], [0, 1])
    pred = clf.predict(vec1)
    print(f"Prediction: {pred}")
