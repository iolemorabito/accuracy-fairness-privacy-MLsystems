from aif360.datasets import CompasDataset, GermanDataset, AdultDataset

def load_dataset(name):
    if name == 'compas':
        return CompasDataset()
    elif name == 'german':
        return GermanDataset()
    elif name == 'adult':
        return AdultDataset()
    else:
        raise ValueError(f"Unknown dataset name: {name}")

