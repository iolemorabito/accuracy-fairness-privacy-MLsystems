import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from art.metrics import PDTP
import diffprivlib.models as dp
from art.estimators.classification import SklearnClassifier
def calculate_pdtp(art_model, X_train, y_train, num_samples):
    """
    Calculate PDTP privacy metric.
    
    Args:
        art_model (object): ART model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        num_samples (int): Number of samples to evaluate.
    
    Returns:
        float: PDTP leakage score.
    """
    sklearn_model =  SklearnClassifier(model=art_model)
    indexes = np.array(range(num_samples))
    leakage, _, _ = PDTP(sklearn_model, sklearn_model, X_train[:num_samples], y_train[:num_samples], indexes=indexes)
    return leakage

def calculate_shapr(model, X_train, y_train, X_test, y_test, num_samples=100):
    """
    Calculate SHAPr privacy metric.
    
    Args:
        model (object): Machine learning model (SklearnClassifier wrapping LogisticRegression, RandomForestClassifier).
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        num_samples (int): Number of samples for Shapley value approximation.
    
    Returns:
        float: SHAPr leakage score.
    """
    # Fit the model with training data
    model.fit(X_train, y_train)

    # Predict test data
    y_pred_test = model.predict(X_test)

    # Access the underlying model from SklearnClassifier
    if isinstance(model, LogisticRegression):
        # Logistic Regression SHAPr
        probs_test = model.predict_proba(X_test)
        shapley_values = np.zeros(probs_test.shape)
        indices = np.random.choice(range(X_train.shape[0]), size=min(num_samples, X_train.shape[0]), replace=False)

        for i in indices:
            X_train_reduced = np.delete(X_train, i, axis=0)
            y_train_reduced = np.delete(y_train, i)
            model.fit(X_train_reduced, y_train_reduced)  # Correct fit call with only two arguments
            probs_test_reduced = model.predict_proba(X_test)
            shapley_values += np.abs(probs_test - probs_test_reduced)

        shapley_values /= len(indices)
        shapr_leakage = np.mean(shapley_values)

    elif isinstance(model, RandomForestClassifier):
        # Random Forest SHAPr
        feature_importances = model.feature_importances_
        num_features = X_train.shape[1]
        shapley_values = np.zeros((X_test.shape[0], num_features))
        indices = np.random.choice(range(X_train.shape[0]), size=min(num_samples, X_train.shape[0]), replace=False)

        for i in indices:
            X_train_reduced = np.delete(X_train, i, axis=0)
            y_train_reduced = np.delete(y_train, i)
            model.fit(X_train_reduced, y_train_reduced)  # Correct fit call with only two arguments
            reduced_importances = model.feature_importances_
            contribution_change = np.abs(feature_importances - reduced_importances)
            shapley_values += contribution_change

        shapley_values /= len(indices)
        shapr_leakage = np.mean(shapley_values)

    else:
        raise ValueError("Model type not supported for SHAPr calculation")

    return shapr_leakage


def calculate_differential_privacy_loss(art_model, X_train, y_train, X_test, y_test, epsilon):
    """
    Calculate differential privacy loss.
    
    Args:
        art_model (object): ART model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        epsilon (float): Privacy budget for differential privacy.
    
    Returns:
        float: Differential privacy loss score.
    """
    try:
        model = art_model.model
        data_norm = np.max(np.linalg.norm(X_train, axis=1))
        
        if isinstance(model, LogisticRegression):
            dp_model = dp.LogisticRegression(epsilon=epsilon, data_norm=data_norm)
        elif isinstance(model, RandomForestClassifier):
            dp_model = dp.RandomForestClassifier(n_samples=100,epsilon=epsilon, data_norm=data_norm)
        else:
            raise ValueError("Model type not supported for differential privacy loss calculation")
        
        dp_model.fit(X_train, y_train)
        dp_score = dp_model.score(X_test, y_test)
        return dp_score

    except Exception as e:
        print(f"Exception during differential privacy loss calculation: {str(e)}")
        return None
