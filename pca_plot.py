import argparse
from datasets.load_datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

   
    pca = PCA()
    X_pca = pca.fit_transform(X_train)
    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Create a 2x1 grid of subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # Plot the explained variance ratio in the first subplot
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Explained Variance Ratio by Principal Component")

    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Plot the cumulative explained variance in the second subplot
    ax2.plot(
        range(1, len(cumulative_explained_variance) + 1),
        cumulative_explained_variance,
        marker="o",
    )
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Explained Variance by Principal Components")

    # Display the figure
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DOE for accuracy, fairness, and privacy testing')
    parser.add_argument('--dataset', type=str, required=True, choices=['compas', 'german', 'adult'], help='Dataset to use')
    #parser.add_argument('--model', type=str, required=True, choices=['logistic_regression', 'random_forest'], help='Model to use')
    
    args = parser.parse_args()

    main(args)