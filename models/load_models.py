from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_model(name):
    if name == 'logistic_regression':
        return LogisticRegression (max_iter=1000, solver='lbfgs')
    elif name == 'random_forest':
        return RandomForestClassifier()
    else:
        raise ValueError(f"Unknown model name: {name}")
