print("MAIN IS STARTING")

import os
from src.loader import load_data
from src.schema_inference import rename_columns
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.train import train_model
from src.predict import make_predictions


def main():
    os.makedirs("outputs", exist_ok=True)

    file_path = "notebook/credit_risk_dataset.csv"
    df = load_data(file_path)

    print("\nRaw data loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    df = rename_columns(df)

    print("\nColumns after renaming:")
    print(list(df.columns))

    df = clean_data(df)

    print("\nData cleaned successfully.")
    print("Shape after cleaning:", df.shape)

    df = engineer_features(df)

    print("\nFeature engineering complete.")
    print("Final shape:", df.shape)

    print("\nPreview of final processed data:")
    print(df.head())

    processed_output_path = "outputs/processed_credit_risk.csv"
    df.to_csv(processed_output_path, index=False)
    print(f"\nProcessed file saved to {processed_output_path}")

    model, X_train, X_test, y_train, y_test = train_model(df)

    print("\nModel trained successfully.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    predictions_df = make_predictions(model, X_test, y_test)

    print("\nPredictions generated successfully.")
    print(predictions_df[["actual", "predicted", "risk_probability", "risk_level"]].head())

    predictions_output_path = "outputs/predictions.csv"
    predictions_df.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions saved to {predictions_output_path}")


if __name__ == "__main__":
    main()