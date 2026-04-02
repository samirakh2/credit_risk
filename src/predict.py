import pandas as pd

def make_predictions(model, X_test, y_test):
    predicted_labels = model.predict(X_test)
    predicted_probs = model.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = predicted_labels
    results["risk_probability"] = predicted_probs

    results["risk_level"] = pd.cut(
        results["risk_probability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return results