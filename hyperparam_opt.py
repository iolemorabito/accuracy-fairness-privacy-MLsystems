import argparse
from datasets.load_datasets import load_dataset
from models.load_models import load_model
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from aif360.datasets import BinaryLabelDataset
from techniques_prova import apply_reweighing, apply_disparate_impact_remover
from metrics.accuracy_metrics import calculate_accuracy_metrics
from metrics.fairness_metrics import calculate_fairness_metrics
from metrics.privacy_metrics import calculate_pdtp, calculate_shapr, calculate_differential_privacy_loss
from art.estimators.classification import SklearnClassifier

# Logistic Regression with Cross-Validation
def logistic_regression_cv_model(X_train, y_train):
    logreg_cv = LogisticRegressionCV(
        Cs=10,            # Number of C values to test
        cv=5,             # Cross-validation folds
        solver='lbfgs',   # Solver for optimization
        scoring='f1',     # Custom scoring metric
        random_state=42,
        max_iter=1000
    )
    
    logreg_cv.fit(X_train, y_train)
    print(f"Best C found: {logreg_cv.C_}")
    return logreg_cv

# Random Forest with Hyperparameter Tuning using GridSearchCV
def random_forest_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def main(args):
    # Load dataset
    try:
        dataset = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        return

    # Dataset-specific settings
    if args.dataset == 'compas':
        privileged_groups = [{'sex': 1, 'race': 1}]
        unprivileged_groups = [{'sex': 0, 'race': 0}]
        protected_attributes = ['sex', 'race']
        label_name = 'two_year_recid'
        favorable_label = 0.0
        unfavorable_label = 1.0
    elif args.dataset == 'german':
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        protected_attributes = ['sex']
        label_name = 'credit'
        favorable_label = 0.0
        unfavorable_label = 1.0
    elif args.dataset == 'adult':
        privileged_groups = [{'race': 1, 'sex': 1}]
        unprivileged_groups = [{'race': 0, 'sex': 0}]
        protected_attributes = ['sex', 'race']
        label_name = 'income-per-year'
        favorable_label = 0.0
        unfavorable_label = 1.0
    else:
        print(f"Dataset '{args.dataset}' is not recognized.")
        return

    try:
        df, _ = dataset.convert_to_dataframe()
    except Exception as e:
        print(f"Error converting dataset to DataFrame: {e}")
        return

    if args.dataset == 'german':
        df[label_name] = df[label_name].map({1.0: favorable_label, 2.0: unfavorable_label})

    binary_label_dataset = BinaryLabelDataset(
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        df=df,
        label_names=[label_name],
        protected_attribute_names=protected_attributes
    )


    X_train, X_test, y_train, y_test = train_test_split(
        binary_label_dataset.features, binary_label_dataset.labels.ravel(), test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Model Selection and Hyperparameter Tuning
    if args.model == 'logistic_regression':
        model = logistic_regression_cv_model(X_train, y_train)
    elif args.model == 'random_forest':
        model = random_forest_grid_search(X_train, y_train)

    # Predict and evaluate the model using your metrics
    y_pred = model.predict(X_test)
    
    # Accuracy Metrics
    accuracy_metrics = calculate_accuracy_metrics(y_test, y_pred)
    print("Accuracy Metrics:", accuracy_metrics)

    # Wrap the model with ART classifier
    art_model = SklearnClassifier(model=model)

    # Fairness Metrics
    fairness_metrics = calculate_fairness_metrics(binary_label_dataset, y_pred, np.arange(len(y_pred)), privileged_groups, unprivileged_groups)
    print("Fairness Metrics:", fairness_metrics)

    # Privacy Metrics
    pdtp_metrics = calculate_pdtp(art_model, X_train, y_train, num_samples=100)
    shapr_metrics = calculate_shapr(model, X_train, y_train, X_test, y_test, num_samples=100)
    differential_privacy_metrics = calculate_differential_privacy_loss(art_model, X_train, y_train, X_test, y_test, epsilon=1.0)

    print("Privacy Metrics:")
    print("PDTP Metric:", np.max(pdtp_metrics))
    print("SHAPr Metrics:", shapr_metrics)
    print("Differential Privacy Loss:", differential_privacy_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DOE for accuracy, fairness, and privacy testing')
    parser.add_argument('--dataset', type=str, required=True, choices=['compas', 'german', 'adult'], help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, choices=['logistic_regression', 'random_forest'], help='Model to use')
    parser.add_argument('--technique', type=str, required=False, choices=['reweighing', 'disparate_impact_remover'], help='Preprocessing technique to use')
    args = parser.parse_args()

    main(args)
