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

    X_train, X_test, y_train, y_test = train_test_split(
        binary_label_dataset.features, binary_label_dataset.labels.ravel(), test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   
    pca = PCA(n_components=0.9,svd_solver='full')

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    try:
        model = load_model(args.model)
        model.fit(X_train_pca, y_train)
    except Exception as e:
        print(f"Error loading or training model '{args.model}': {e}")
        return

    if not hasattr(model, 'classes_'):
        raise ValueError(f"The model {args.model} is not fitted yet.")

    # Wrap the model with ART classifier
    art_model = SklearnClassifier(model=model)

    # Make predictions on the PCA-transformed test data
    y_pred = model.predict(X_test_pca)

    try:   
        accuracy_metrics = calculate_accuracy_metrics(y_test, y_pred)
        fairness_metrics = calculate_fairness_metrics(binary_label_dataset, y_pred, np.arange(len(y_pred)), privileged_groups, unprivileged_groups)
        pdtp_metrics = calculate_pdtp(art_model, X_test_pca, y_test, num_samples=100)
        shapr_metrics = calculate_shapr(model, X_train_pca, y_train, X_test_pca, y_test, num_samples=100)
        differential_privacy_metrics = calculate_differential_privacy_loss(art_model, X_train_pca, y_train, X_test_pca, y_test, epsilon=1.0)

        metrics = {
            'Accuracy Metrics': accuracy_metrics,
            'Fairness Metrics': fairness_metrics,
            'Max PDTP Metric': np.max(pdtp_metrics),
            'SHAPr Metrics': shapr_metrics,
            'Differential Privacy Metrics': differential_privacy_metrics
        }

        for category, values in metrics.items():
            if isinstance(values, dict):
                for value in values.values():
                    print(value)
            else:
                print(values)
    except Exception as e:
        print(f"Error calculating metrics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DOE for accuracy, fairness, and privacy testing')
    parser.add_argument('--dataset', type=str, required=True, choices=['compas', 'german', 'adult'], help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, choices=['logistic_regression', 'random_forest'], help='Model to use')
    parser.add_argument('--technique', type=str, required=False, choices=['membershipinferenceattack','reweighing','disparate_impact_remover','grid_search_reduction','exponentiated_gradient_reduction','calibrated_eq_odds','eq_odds_postprocessing','prejudice_remover','gaussian_mechanism','laplace_mechanism','anonymization','pca'], help='Model to use')
    args = parser.parse_args()

    main(args)