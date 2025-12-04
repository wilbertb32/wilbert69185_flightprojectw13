import joblib

def predict(data):
    clf = joblib.load("rfmodel.sav")
    return clf.predict(data)