import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title="Smart Grid Agentic AI",
    layout="wide"
)

st.title("⚡ Smart Grid Agentic AI System")
st.write("Ask questions about energy generation, forecasting, anomalies, dates, and renewable energy.")

# =========================
# LOAD DATA
# =========================

file_path = "Smart_Grid_EIA930_3Years_Modified_Venkatesh_Kota.xlsx"

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna()
df = df.sort_values("Date")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df["Prev_Day_Total"] = df["Total_Generation_MW"].shift(1)
df["Prev_Day_Solar"] = df["Solar_MW"].shift(1)
df["Prev_Day_Wind"] = df["Wind_MW"].shift(1)

df = df.dropna()

# =========================
# ML MODEL
# =========================

features = [
    "Prev_Day_Total",
    "Prev_Day_Solar",
    "Prev_Day_Wind",
    "Month"
]

X = df[features]
y = df["Total_Generation_MW"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

baseline_pred = np.full(len(y_test), y_train.mean())

results = pd.DataFrame({
    "Model": ["Baseline Average", "Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, baseline_pred),
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, baseline_pred)),
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ],
    "R2 Score": [
        r2_score(y_test, baseline_pred),
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred)
    ]
})

# =========================
# ANOMALY MODEL
# =========================

iso_model = IsolationForest(contamination=0.03, random_state=42)
df["Anomaly"] = iso_model.fit_predict(df[["Total_Generation_MW"]])
df["Anomaly_Label"] = df["Anomaly"].map({-1: "Anomaly", 1: "Normal"})

# =========================
# AGENT DATA LOGIC
# =========================

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
        return pd.to_datetime(date_match.group(0), errors="coerce")

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
            "Tool Selected: Date-Based Data Lookup Tool\n\n"
            f"{display_names[column]} on {date_value.date()} was "
            f"{round(value, 2)} {unit}."
        )

    # Maximum
    if column is not None and (
        "max" in query or "maximum" in query or "highest" in query or "peak" in query
    ):
        row = df.loc[df[column].idxmax()]
        value = row[column]
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            "Tool Selected: Maximum Value Lookup Tool\n\n"
            f"The maximum {display_names[column]} occurred on {row['Date'].date()}.\n"
            f"Value: {round(value, 2)} {unit}"
        )

    # Minimum
    if column is not None and (
        "min" in query or "minimum" in query or "lowest" in query
    ):
        row = df.loc[df[column].idxmin()]
        value = row[column]
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            "Tool Selected: Minimum Value Lookup Tool\n\n"
            f"The minimum {display_names[column]} occurred on {row['Date'].date()}.\n"
            f"Value: {round(value, 2)} {unit}"
        )

    # Average
    if column is not None and (
        "average" in query or "avg" in query or "mean" in query
    ):
        value = df[column].mean()
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            "Tool Selected: Average Analysis Tool\n\n"
            f"The average {display_names[column]} is {round(value, 2)} {unit}."
        )

    # General summary
    if column is not None:
        avg = df[column].mean()
        max_value = df[column].max()
        min_value = df[column].min()
        unit = "%" if column == "Renewable_Share_%" else "MW"

        return (
            "Tool Selected: General Data Analysis Tool\n\n"
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
            if source in query and source != "renewable" and source != "total":
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

    # Dataset lookup
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

# =========================
# USER INTERFACE
# =========================

st.subheader("Ask the Smart Grid Agent")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Example: Give me 01/03/2023 solar generation"
)

if st.button("Ask Agent"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer = smart_grid_agent(user_question)
        st.success("Agent Response")
        st.text(answer)

# =========================
# SAMPLE QUESTIONS
# =========================

st.subheader("Example Questions")

sample_questions = [
    "Give me 01/03/2023 solar generation",
    "Give me 01/03/2023 wind generation",
    "What is the date of maximum solar generation?",
    "When was wind generation lowest?",
    "Average coal generation",
    "Compare solar and wind",
    "Predict next energy generation",
    "Detect abnormal generation",
    "Explain energy trend"
]

st.write(sample_questions)

# =========================
# DISPLAY RESULTS
# =========================

st.subheader("Model Evaluation Results")
st.dataframe(results)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Total Energy Generation Over Time")
st.line_chart(df.set_index("Date")["Total_Generation_MW"])
