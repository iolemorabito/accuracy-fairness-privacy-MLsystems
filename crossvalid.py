import argparse
from datasets.load_datasets import load_dataset
from models.load_models import load_model
from metrics.accuracy_metrics import calculate_accuracy_metrics
from metrics.fairness_metrics import calculate_fairness_metrics
from metrics.privacy_metrics import calculate_pdtp, calculate_shapr ,calculate_differential_privacy_loss
from sklearn.model_selection import train_test_split
import numpy as np
from art.estimators.classification import SklearnClassifier
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold

def main(args):
    # Load dataset
    try:
        dataset = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        return

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

     # Initialize KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Placeholder for metrics
    accuracy_scores = {
        'balanced_accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    fairness_scores = {
        'statistical_parity_difference': [],
        'equal_opportunity_difference': [],
        'average_odds_difference': []
    }
    
    privacy_scores = {
        'pdtp': [],
        'shapr': [],
        'differential_privacy_loss': []
    }


    # Cross-validation loop
    for train_index, test_index in kf.split(binary_label_dataset.features):
        X_train, X_test = binary_label_dataset.features[train_index], binary_label_dataset.features[test_index]
        y_train, y_test = binary_label_dataset.labels.ravel()[train_index], binary_label_dataset.labels.ravel()[test_index]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    try:
        model = load_model(args.model)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error loading or training model '{args.model}': {e}")
        return

    if not hasattr(model, 'classes_'):
        raise ValueError(f"The model {args.model} is not fitted yet.")

    # Wrap the model with ART classifier
    art_model = SklearnClassifier(model=model)

    # Make predictions on the PCA-transformed test data
    y_pred = model.predict(X_test)
    try:   
       # Calculate Accuracy Metrics
        accuracy_metrics = calculate_accuracy_metrics(y_test, y_pred)
        accuracy_scores['balanced_accuracy'].append(accuracy_metrics['balanced_accuracy'])
        accuracy_scores['precision'].append(accuracy_metrics['precision'])
        accuracy_scores['recall'].append(accuracy_metrics['recall'])
        accuracy_scores['f1_score'].append(accuracy_metrics['f1_score'])
        
        # Calculate Fairness Metrics
        fairness_metrics = calculate_fairness_metrics(binary_label_dataset, y_pred, test_index, privileged_groups, unprivileged_groups)
        fairness_scores['statistical_parity_difference'].append(fairness_metrics['statistical_parity_difference'])
        fairness_scores['equal_opportunity_difference'].append(fairness_metrics['equal_opportunity_difference'])
        fairness_scores['average_odds_difference'].append(fairness_metrics['average_odds_difference'])
        
        # Calculate Privacy Metrics
        pdtp_metrics = calculate_pdtp(SklearnClassifier(model), X_train, y_train, num_samples=100)
        shapr_metrics = calculate_shapr(model, X_train, y_train, X_test, y_test, num_samples=100)
        differential_privacy_metrics = calculate_differential_privacy_loss(SklearnClassifier(model), X_train, y_train, X_test, y_test, epsilon=1.0)

        privacy_scores['pdtp'].append(np.max(pdtp_metrics))
        privacy_scores['shapr'].append(shapr_metrics)
        privacy_scores['differential_privacy_loss'].append(differential_privacy_metrics)

        # Aggregate results after cross-validation
        print(np.mean(accuracy_scores['balanced_accuracy']))
        print(np.mean(accuracy_scores['precision']))
        print(np.mean(accuracy_scores['recall']))
        print(np.mean(accuracy_scores['f1_score']))

        print(np.mean(fairness_scores['statistical_parity_difference']))
        print(np.mean(fairness_scores['equal_opportunity_difference']))
        print(np.mean(fairness_scores['average_odds_difference']))

        print(np.mean(privacy_scores['pdtp']))
        print(np.mean(privacy_scores['shapr']))
        print(np.mean(privacy_scores['differential_privacy_loss']))
    except Exception as e:
        print(f"Error calculating metrics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DOE for accuracy, fairness, and privacy testing')
    parser.add_argument('--dataset', type=str, required=True, choices=['compas', 'german', 'adult'], help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, choices=['logistic_regression', 'random_forest'], help='Model to use')
    parser.add_argument('--technique', type=str, required=False, choices=['membershipinferenceattack','reweighing','disparate_impact_remover','grid_search_reduction','exponentiated_gradient_reduction','calibrated_eq_odds','eq_odds_postprocessing','prejudice_remover','gaussian_mechanism','laplace_mechanism','anonymization','pca'], help='Model to use')
    args = parser.parse_args()

    main(args)