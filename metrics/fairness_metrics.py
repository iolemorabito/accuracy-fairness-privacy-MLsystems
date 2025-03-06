from aif360.metrics import ClassificationMetric
import pandas as pd

def calculate_fairness_metrics(dataset, y_pred, y_test_indices, privileged_groups, unprivileged_groups):
    """
    Calculate fairness metrics given the dataset and predictions.
    
    Args:
        dataset (BinaryLabelDataset): AIF360 dataset.
        y_pred (np.ndarray): Predicted labels.
        y_test_indices (np.ndarray): Indices of the test set.
        privileged_groups (list): Privileged groups for fairness evaluation.
        unprivileged_groups (list): Unprivileged groups for fairness evaluation.

    Returns:
        dict: Dictionary containing fairness metrics.
    """
    # Filter the dataset to include only the test instances
    test_dataset = dataset.subset(y_test_indices)
    
    # Ensure labels in the pred_dataset match y_pred
    pred_dataset = test_dataset.copy(deepcopy=True)
    pred_dataset.labels = y_pred

    metric_true = ClassificationMetric(
        test_dataset, pred_dataset,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )

    # Calculate fairness metrics
    fairness_metrics = {
        'statistical_parity_difference': metric_true.statistical_parity_difference(),
        'equal_opportunity_difference': metric_true.equal_opportunity_difference(),
        'average_odds_difference': metric_true.average_odds_difference(),
    }

    return fairness_metrics
