from sklearn.metrics import accuracy_score

# TODO make sklearn work when this is unpicklable
def score_func(y, y_pred):
    acc = accuracy_score(y, y_pred)
    return abs(acc - 0.95)