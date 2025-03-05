from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
from aif360.algorithms.inprocessing import GridSearchReduction, ExponentiatedGradientReduction
from models.load_models import load_model
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
from diffprivlib.mechanisms import Gaussian, Laplace
from apt.anonymization.anonymizer import Anonymize
from apt.utils.datasets.datasets import ArrayDataset
from aif360.datasets import BinaryLabelDataset
import pandas as pd
import numpy as np

#PRE-PROCESSING
#OK
def apply_disparate_impact_remover(dataset, repair_level=1.0, sensitive_attribute=''):
    dir = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=sensitive_attribute)
    return dir.fit_transform(dataset)

#OK
def apply_reweighing(dataset, unprivileged_groups, privileged_groups):
    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    rw.fit(dataset)
    return rw.transform(dataset)



#IN-PROCESSING

#OK
def apply_grid_search_reduction(dataset, model_name, constraints, prot_attr, constraint_weight=0.5, grid_size=10, grid_limit=2.0, drop_prot_attr=True,loss='ZeroOne'):
    # Load the model
    estimator = load_model(model_name)
    
    # Define the constraints
    if constraints == 'DemographicParity':
        constraints_obj = 'DemographicParity'
    elif constraints == 'EqualizedOdds':
        constraints_obj = 'EqualizedOdds'
    else:
        raise ValueError("Unsupported constraint type")

    # Prepare the GridSearchReduction object
    grid_search = GridSearchReduction(
        estimator=estimator,
        constraints=constraints_obj,
        prot_attr=prot_attr,
        constraint_weight=constraint_weight,
          grid_size=grid_size,
        grid_limit=grid_limit,
        drop_prot_attr=drop_prot_attr,
        loss=loss
    )

    # Fit the GridSearchReduction model with the AIF360 dataset
    grid_search.fit(dataset)
    
    transformed_dataset = grid_search.predict(dataset)
    return transformed_dataset

def apply_exponentiated_gradient_reduction(dataset, model_name, constraints='DemographicParity', eps=0.01, max_iter=50, nu=None, eta0=2.0, run_linprog_step=True, drop_prot_attr=True):
    # Load the model
    estimator = load_model(model_name)
    
    # Prepare the ExponentiatedGradientReduction object
    exp_grad_red = ExponentiatedGradientReduction(
        estimator=estimator,
        constraints=constraints,
        eps=eps,
        max_iter=max_iter,
        nu=nu,
        eta0=eta0,
        run_linprog_step=run_linprog_step,
        drop_prot_attr=drop_prot_attr
    )

    # Fit the ExponentiatedGradientReduction model with the AIF360 dataset
    exp_grad_red.fit(dataset)
    
    # Predict using the fitted model
    transformed_dataset = exp_grad_red.predict(dataset)
    
    return transformed_dataset


#POST-PROCESSING
#OK
def apply_calibrated_eq_odds(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    # Initialize the post-processing object
    cpp = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        cost_constraint="fnr",
        seed=12345679  #Seed to make predict repeatable.
    )
    
    # Fit the post-processing model
    cpp = cpp.fit(dataset_true, dataset_pred)
    
    # Apply the transformation and return the result
    transformed_dataset = cpp.predict(dataset_pred)
    
    return transformed_dataset

#OK
def apply_eq_odds_postprocessing(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    # Initialize the post-processing object
    eop = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        seed=12345679  # Seed to make predict repeatable
    )
    
    # Fit the post-processing model
    eop = eop.fit(dataset_true, dataset_pred)
    
    # Apply the transformation and return the result
    transformed_dataset = eop.predict(dataset_pred)
    
    return transformed_dataset

#PRIVACY TECHNIQUES
#OK
def apply_gaussian_mechanism_to_features(dataset, epsilon, delta, sensitivity, random_state):
    gaussian_mechanism = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
    for i in range(dataset.features.shape[0]):
        for j in range(dataset.features.shape[1]):
            dataset.features[i, j] = gaussian_mechanism.randomise(dataset.features[i, j])
    return dataset

# Laplace mechanism to add noise to dataset features
def apply_laplace_mechanism_to_features(dataset, epsilon, sensitivity, random_state):
    laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity, random_state=random_state)
    for i in range(dataset.features.shape[0]):
        for j in range(dataset.features.shape[1]):
            dataset.features[i, j] = laplace_mechanism.randomise(dataset.features[i, j])
    return dataset

def apply_anonymization(binary_label_dataset, k, quasi_identifiers):
    # Convert BinaryLabelDataset to a DataFrame
    df, _ = binary_label_dataset.convert_to_dataframe()
    
    # Get the feature names from the DataFrame
    feature_names = list(df.columns)
    
    # Debugging: print the feature names and quasi identifiers
    print(f"Feature Names: {feature_names}")
    print(f"Quasi-Identifiers: {quasi_identifiers}")

    # Ensure quasi_identifiers are a subset of feature names
    for identifier in quasi_identifiers:
        if identifier not in feature_names:
            raise ValueError(f"Quasi-identifier '{identifier}' is not a valid feature name. Valid feature names are: {feature_names}")

    # Convert to ArrayDataset
    X = df.values
    y = df[binary_label_dataset.label_names].values

    # Get the indices of quasi-identifiers
    quasi_identifier_indices = [feature_names.index(identifier) for identifier in quasi_identifiers]

    # Debugging: print the indices of quasi-identifiers
    print(f"Quasi-Identifier Indices: {quasi_identifier_indices}")

    # Verify the dimensions of X
    print(f"X shape: {X.shape}")

    # Ensure quasi-identifier index is within the range
    for idx in quasi_identifier_indices:
        if idx >= X.shape[1]:
            raise ValueError(f"Quasi-identifier index {idx} is out of range. The dataset has {X.shape[1]} columns.")

    # Create ArrayDataset
    array_dataset = ArrayDataset(x=X, y=y, features_names=feature_names)

    # Verify the ArrayDataset structure before anonymization
    print(f"ArrayDataset Features: {array_dataset.features_names}")
  

    # Apply anonymization
    anonymizer = Anonymize(k=k, quasi_identifiers=quasi_identifier_indices)
    anonymized_data = anonymizer.anonymize(array_dataset)

    # Convert anonymized data back to DataFrame
    anonymized_df = pd.DataFrame(anonymized_data, columns=feature_names)

    # Convert back to BinaryLabelDataset
    binary_label_dataset = BinaryLabelDataset(
        favorable_label=binary_label_dataset.favorable_label,
        unfavorable_label=binary_label_dataset.unfavorable_label,
        df=anonymized_df,
        label_names=binary_label_dataset.label_names,
        protected_attribute_names=quasi_identifiers
    )

    return binary_label_dataset

