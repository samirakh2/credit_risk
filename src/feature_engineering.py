def engineer_features(df):

    # already partially given but still useful
    df["debt_to_income"] = df["loan_amount"] / df["income"].replace(0, 1)

    # risk based on employment
    df["income_per_year_employed"] = df["income"] / (df["employment_length"] + 1)

    # credit maturity
    df["credit_history_ratio"] = df["credit_history_length"] / df["age"]

    # high risk flag
    df["high_loan_percent"] = (df["loan_percent_income"] > 0.4).astype(int)

    return df