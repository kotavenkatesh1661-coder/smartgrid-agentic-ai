import re

energy_sources = {
    "solar": "Solar_MW",
    "wind": "Wind_MW",
    "hydro": "Hydro_MW",
    "coal": "Coal_MW",
    "gas": "Gas_MW",
    "nuclear": "Nuclear_MW",
    "total": "Total_Generation_MW",
    "renewable": "Renewable_Share_%"
}

display_names = {
    "Solar_MW": "Solar Generation",
    "Wind_MW": "Wind Generation",
    "Hydro_MW": "Hydro Generation",
    "Coal_MW": "Coal Generation",
    "Gas_MW": "Gas Generation",
    "Nuclear_MW": "Nuclear Generation",
    "Total_Generation_MW": "Total Energy Generation",
    "Renewable_Share_%": "Renewable Share"
}


def find_energy_column(query):
    for source, column in energy_sources.items():
        if source in query:
            return column
    return None


def find_date(query):
    date_match = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", query)

    if date_match:
        try:
            return pd.to_datetime(date_match.group(0), errors="coerce")
        except:
            return None

    return None


def answer_from_dataset(query):
    query = query.lower()

    column = find_energy_column(query)
    date_value = find_date(query)

    # Date-based lookup
    if date_value is not None and column is not None:
        matching_row = df[df["Date"].dt.date == date_value.date()]

        if matching_row.empty:
            return f"No data found for {date_value.date()}."

        row = matching_row.iloc[0]
        value = row[column]

        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            f"Tool Selected: Date-Based Data Lookup Tool\n\n"
            f"{display_names[column]} on {date_value.date()} was "
            f"{round(value, 2)} {unit}."
        )

    # Max / highest / peak
    if column is not None and (
        "max" in query or "maximum" in query or "highest" in query or "peak" in query
    ):
        row = df.loc[df[column].idxmax()]
        value = row[column]
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            f"Tool Selected: Maximum Value Lookup Tool\n\n"
            f"The maximum {display_names[column]} occurred on {row['Date'].date()}.\n"
            f"Value: {round(value, 2)} {unit}"
        )

    # Min / lowest
    if column is not None and (
        "min" in query or "minimum" in query or "lowest" in query
    ):
        row = df.loc[df[column].idxmin()]
        value = row[column]
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            f"Tool Selected: Minimum Value Lookup Tool\n\n"
            f"The minimum {display_names[column]} occurred on {row['Date'].date()}.\n"
            f"Value: {round(value, 2)} {unit}"
        )

    # Average / mean
    if column is not None and (
        "average" in query or "avg" in query or "mean" in query
    ):
        value = df[column].mean()
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            f"Tool Selected: Average Analysis Tool\n\n"
            f"The average {display_names[column]} is {round(value, 2)} {unit}."
        )

    # General source summary
    if column is not None:
        avg = df[column].mean()
        max_value = df[column].max()
        min_value = df[column].min()
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            f"Tool Selected: General Data Analysis Tool\n\n"
            f"{display_names[column]} Summary:\n"
            f"- Average: {round(avg, 2)} {unit}\n"
            f"- Maximum: {round(max_value, 2)} {unit}\n"
            f"- Minimum: {round(min_value, 2)} {unit}"
        )

    return None


def smart_grid_agent(user_query):
    query = user_query.lower()

    forecast_keywords = ["forecast", "predict", "future", "estimate", "next"]
    anomaly_keywords = ["anomaly", "abnormal", "unusual", "spike", "drop"]
    compare_keywords = ["compare", "versus", "vs"]
    trend_keywords = ["trend", "pattern", "change over time", "behavior"]

    # Forecasting
    if any(word in query for word in forecast_keywords):
        pred = lr_model.predict(X_test.iloc[-1:])[0]

        return (
            "Tool Selected: ML Forecasting Tool\n\n"
            f"Predicted next total energy generation: {round(pred, 2)} MW\n\n"
            "Model Used: Linear Regression"
        )

    # Anomaly detection
    if any(word in query for word in anomaly_keywords):
        anomaly_count = len(df[df["Anomaly_Label"] == "Anomaly"])

        return (
            "Tool Selected: Anomaly Detection Tool\n\n"
            f"The system detected {anomaly_count} abnormal energy generation events.\n\n"
            "Model Used: Isolation Forest"
        )

    # Comparison
    if any(word in query for word in compare_keywords):
        mentioned_sources = []

        for source, column in energy_sources.items():
            if source in query and source != "renewable":
                mentioned_sources.append((source, column))

        if len(mentioned_sources) >= 2:
            source1, col1 = mentioned_sources[0]
            source2, col2 = mentioned_sources[1]

            avg1 = df[col1].mean()
            avg2 = df[col2].mean()

            if avg1 > avg2:
                conclusion = f"{source1.capitalize()} has higher average generation."
            elif avg2 > avg1:
                conclusion = f"{source2.capitalize()} has higher average generation."
            else:
                conclusion = "Both sources have similar average generation."

            return (
                "Tool Selected: Comparative Analysis Tool\n\n"
                f"- {source1.capitalize()} Average: {round(avg1, 2)} MW\n"
                f"- {source2.capitalize()} Average: {round(avg2, 2)} MW\n"
                f"- Conclusion: {conclusion}"
            )

    # Dataset question handler
    dataset_answer = answer_from_dataset(query)

    if dataset_answer is not None:
        return dataset_answer

    # Trend explanation
    if any(word in query for word in trend_keywords):
        return (
            "Tool Selected: Trend Analysis Tool\n\n"
            "The smart grid dataset shows time-based variation in solar, wind, hydro, "
            "coal, gas, nuclear, renewable share, and total generation."
        )

    return (
        "I can answer questions from the smart grid dataset about solar, wind, hydro, "
        "coal, gas, nuclear, total generation, renewable share, dates, maximum values, "
        "minimum values, averages, comparisons, forecasting, and anomalies."
    )
