import argparse
from datasets.load_datasets import load_dataset
from models.load_models import load_model
from metrics.accuracy_metrics import calculate_accuracy_metrics
from metrics.fairness_metrics import calculate_fairness_metrics
from metrics.privacy_metrics import calculate_pdtp, calculate_shapr ,calculate_differential_privacy_loss
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from sklearn.model_selection import train_test_split
import numpy as np
from art.estimators.classification import SklearnClassifier
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from apt.anonymization.anonymizer import Anonymize
from apt.utils.datasets import ArrayDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing,EqOddsPostprocessing
from techniques import  apply_reweighing,apply_grid_search_reduction, apply_exponentiated_gradient_reduction, apply_disparate_impact_remover, apply_gaussian_mechanism_to_features,apply_laplace_mechanism_to_features

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
        favorable_label = 0.0 #1.0
        unfavorable_label = 1.0 #2.0
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

    #PRE-PROCESSING TECHNIQUES
    if args.technique == 'reweighing':
        binary_label_dataset= apply_reweighing(binary_label_dataset, privileged_groups, unprivileged_groups)
    elif args.technique == 'disparate_impact_remover':
            binary_label_dataset = apply_disparate_impact_remover(binary_label_dataset, repair_level=1.0, sensitive_attribute=protected_attributes[0])
    elif args.technique == 'gaussian_mechanism':
        binary_label_dataset = apply_gaussian_mechanism_to_features(binary_label_dataset, epsilon=0.5, delta=1e-5, sensitivity=1.0, random_state=42)
    elif args.technique == 'laplace_mechanism':
        binary_label_dataset = apply_laplace_mechanism_to_features(binary_label_dataset, epsilon=0.5, sensitivity=1.0, random_state=42)


    X_train, X_test, y_train, y_test = train_test_split(
        binary_label_dataset.features, binary_label_dataset.labels.ravel(), test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #IN PROCESSING
    if args.technique == 'grid_search_reduction':
        binary_label_dataset = apply_grid_search_reduction(
        dataset=binary_label_dataset, model_name=args.model,
        constraints='DemographicParity',  # or 'EqualizedOdds'
        prot_attr=protected_attributes[1],  constraint_weight=0.5, grid_size=10,
        grid_limit=2.0, drop_prot_attr=True, loss='ZeroOne'
    )        
    elif args.technique == 'exponentiated_gradient_reduction':
         binary_label_dataset = apply_exponentiated_gradient_reduction(
        dataset=binary_label_dataset, model_name=args.model,
        constraints='DemographicParity',  # or 'EqualizedOdds'
        eps=0.01, max_iter=50, nu=None, eta0=2.0,
        run_linprog_step=True, drop_prot_attr=True
    )
    elif args.technique == 'anonymization':
        if args.dataset == 'adult':
            quasi_identifiers=[0,4,10,12]
        elif args.dataset == 'german':
            quasi_identifiers = [1, 2, 3, 9, 11, 13,14, 6]
        elif args.dataset == 'compas':
            quasi_identifiers = [18,25,34]
        try:
            array_dataset = ArrayDataset(X_train, y_train)
            anonymizer = Anonymize(
                k=50,
                quasi_identifiers=quasi_identifiers
            )
            X_train_anonymized = anonymizer.anonymize(array_dataset)
            X_train = X_train_anonymized if isinstance(X_train_anonymized, np.ndarray) else X_train_anonymized.values
        except Exception as e:
            print(f"Error during anonymization: {e}")
            return   
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

    y_pred = model.predict(X_test)

    #POST-PROCESSING TECHNIQUES
    if args.technique == 'calibrated_eq_odds':
        dataset_orig_train, dataset_orig_vt = binary_label_dataset.split([0.6], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

        cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                            unprivileged_groups = unprivileged_groups,
                                            cost_constraint="fnr",
                                            seed=12345679 #Seed to make predict repeatable.
                                            )
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
        binary_label_dataset = cpp.predict (dataset_orig_test_pred)

    elif args.technique == 'eq_odds_postprocessing':
        dataset_orig_train, dataset_orig_vt = binary_label_dataset.split([0.6], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        eop = EqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            seed=12345679  # Seed to make predict repeatable
        )
        
        eop = eop.fit(dataset_orig_valid, dataset_orig_valid_pred)

        binary_label_dataset = eop.predict (dataset_orig_test_pred)

    elif args.technique == 'membershipinferenceattack':
        # Perform the Membership Inference Black-Box attack
        attack = MembershipInferenceBlackBox(estimator=art_model)
        # Fit the attack model
        attack.fit(x=X_train, y=y_train, test_x=X_test, test_y=y_test)
        # Infer membership
        membership_inference_results = attack.infer(x=X_test, y=y_test)

        print("Membership Inference Results:")
        print(membership_inference_results)

    try:   

        accuracy_metrics = calculate_accuracy_metrics(y_test, y_pred)
        fairness_metrics = calculate_fairness_metrics(binary_label_dataset, y_pred, np.arange(len(y_pred)), privileged_groups, unprivileged_groups)
        pdtp_metrics = calculate_pdtp(art_model, X_train, y_train, num_samples=100)
        shapr_metrics = calculate_shapr(model, X_train, y_train, X_test, y_test, num_samples=100)
        differential_privacy_metrics = calculate_differential_privacy_loss(art_model, X_train, y_train, X_test, y_test, epsilon=1.0)

        
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
    parser.add_argument('--technique', type=str, required=False, choices=['membershipinferenceattack','reweighing','disparate_impact_remover','grid_search_reduction','exponentiated_gradient_reduction','calibrated_eq_odds','eq_odds_postprocessing','prejudice_remover','gaussian_mechanism','laplace_mechanism','anonymization'], help='Model to use')
    args = parser.parse_args()

    main(args)