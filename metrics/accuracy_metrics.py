from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy_metrics(y_true, y_pred):

    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics
