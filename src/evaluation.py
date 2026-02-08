from sklearn.metrics import classification_report, confusion_matrix


def report_metrics(y_true, y_pred, labels=None):
    print(classification_report(y_true, y_pred, labels=labels))
    print(confusion_matrix(y_true, y_pred))
