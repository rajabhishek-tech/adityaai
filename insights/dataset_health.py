def calculate_dataset_health(health_df):
    score = 100

    for _, row in health_df.iterrows():
        if "🔴" in row["Health"]:
            score -= 15

        if "identifier" in row["Issues"].lower():
            score -= 10

        if "missing" in row["Issues"].lower():
            score -= 20

        if "outlier" in row["Issues"].lower():
            score -= 10

    return max(score, 0)
