import streamlit as st
import pandas as pd
import numpy as np

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
st.write("Ask questions about energy generation, forecasting, anomalies, and renewable energy.")

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
# ML FORECASTING MODEL
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
# ANOMALY DETECTION
# =========================

iso_model = IsolationForest(contamination=0.03, random_state=42)
df["Anomaly"] = iso_model.fit_predict(df[["Total_Generation_MW"]])
df["Anomaly_Label"] = df["Anomaly"].map({-1: "Anomaly", 1: "Normal"})

# =========================
# AGENT FUNCTIONS
# =========================

def contains_any(query, keywords):
    return any(word in query for word in keywords)


def get_average_generation(source_column, source_name):
    avg = df[source_column].mean()
    max_value = df[source_column].max()
    min_value = df[source_column].min()

    return (
        f"Tool Selected: {source_name} Analysis Tool\n\n"
        f"{source_name} Generation Analysis:\n"
        f"- Average: {round(avg, 2)} MW\n"
        f"- Maximum: {round(max_value, 2)} MW\n"
        f"- Minimum: {round(min_value, 2)} MW"
    )


def get_max_min_date(source_column, source_name, query):
    if "max" in query or "maximum" in query or "highest" in query or "peak" in query:
        row = df.loc[df[source_column].idxmax()]
        return (
            f"Tool Selected: {source_name} Maximum Date Tool\n\n"
            f"The maximum {source_name.lower()} generation occurred on {row['Date'].date()}.\n"
            f"Generation Value: {round(row[source_column], 2)} MW"
        )

    if "min" in query or "minimum" in query or "lowest" in query:
        row = df.loc[df[source_column].idxmin()]
        return (
            f"Tool Selected: {source_name} Minimum Date Tool\n\n"
            f"The minimum {source_name.lower()} generation occurred on {row['Date'].date()}.\n"
            f"Generation Value: {round(row[source_column], 2)} MW"
        )

    return None


def compare_sources(source1_col, source1_name, source2_col, source2_name):
    avg1 = df[source1_col].mean()
    avg2 = df[source2_col].mean()

    if avg1 > avg2:
        conclusion = f"{source1_name} has higher average generation than {source2_name}."
    elif avg2 > avg1:
        conclusion = f"{source2_name} has higher average generation than {source1_name}."
    else:
        conclusion = f"{source1_name} and {source2_name} have similar average generation."

    return (
        "Tool Selected: Comparative Analysis Tool\n\n"
        f"- {source1_name} Average: {round(avg1, 2)} MW\n"
        f"- {source2_name} Average: {round(avg2, 2)} MW\n"
        f"- Conclusion: {conclusion}"
    )


def smart_grid_agent(user_query):
    query = user_query.lower()

    forecast_keywords = ["forecast", "predict", "future", "estimate", "next"]
    anomaly_keywords = ["anomaly", "abnormal", "unusual", "spike", "drop"]
    trend_keywords = ["trend", "pattern", "change over time", "behavior"]
    renewable_keywords = ["renewable", "renewable share", "clean energy"]
    compare_keywords = ["compare", "versus", "vs", "higher", "more"]

    energy_sources = {
        "solar": ("Solar_MW", "Solar"),
        "wind": ("Wind_MW", "Wind"),
        "hydro": ("Hydro_MW", "Hydro"),
        "coal": ("Coal_MW", "Coal"),
        "gas": ("Gas_MW", "Gas"),
        "nuclear": ("Nuclear_MW", "Nuclear"),
        "total": ("Total_Generation_MW", "Total Energy")
    }

    # Forecasting tool
    if contains_any(query, forecast_keywords):
        pred = lr_model.predict(X_test.iloc[-1:])[0]
        return (
            "Tool Selected: ML Forecasting Tool\n\n"
            f"Predicted next total energy generation: {round(pred, 2)} MW\n\n"
            "Model Used: Linear Regression"
        )

    # Anomaly detection tool
    elif contains_any(query, anomaly_keywords):
        anomaly_count = len(df[df["Anomaly_Label"] == "Anomaly"])
        return (
            "Tool Selected: Anomaly Detection Tool\n\n"
            f"The system detected {anomaly_count} abnormal energy generation events.\n\n"
            "Model Used: Isolation Forest"
        )

    # Maximum / minimum date questions
    for source_key, (column_name, display_name) in energy_sources.items():
        if source_key in query:
            max_min_response = get_max_min_date(column_name, display_name, query)
            if max_min_response is not None:
                return max_min_response

    # Comparison tool
    if contains_any(query, compare_keywords):
        if "solar" in query and "wind" in query:
            return compare_sources("Solar_MW", "Solar", "Wind_MW", "Wind")
        elif "coal" in query and "gas" in query:
            return compare_sources("Coal_MW", "Coal", "Gas_MW", "Gas")
        elif "hydro" in query and "nuclear" in query:
            return compare_sources("Hydro_MW", "Hydro", "Nuclear_MW", "Nuclear")
        else:
            source_means = {
                "Solar": df["Solar_MW"].mean(),
                "Wind": df["Wind_MW"].mean(),
                "Hydro": df["Hydro_MW"].mean(),
                "Coal": df["Coal_MW"].mean(),
                "Gas": df["Gas_MW"].mean(),
                "Nuclear": df["Nuclear_MW"].mean()
            }

            highest_source = max(source_means, key=source_means.get)

            return (
                "Tool Selected: Comparative Analysis Tool\n\n"
                f"The highest average generation source is {highest_source} "
                f"with {round(source_means[highest_source], 2)} MW."
            )

    # Renewable share
    elif contains_any(query, renewable_keywords):
        avg = df["Renewable_Share_%"].mean()
        max_value = df["Renewable_Share_%"].max()
        min_value = df["Renewable_Share_%"].min()

        return (
            "Tool Selected: Renewable Energy Analysis Tool\n\n"
            f"- Average Renewable Share: {round(avg, 2)}%\n"
            f"- Maximum Renewable Share: {round(max_value, 2)}%\n"
            f"- Minimum Renewable Share: {round(min_value, 2)}%"
        )

    # Source-specific analysis tools
    elif "solar" in query:
        return get_average_generation("Solar_MW", "Solar")

    elif "wind" in query:
        return get_average_generation("Wind_MW", "Wind")

    elif "hydro" in query:
        return get_average_generation("Hydro_MW", "Hydro")

    elif "coal" in query:
        return get_average_generation("Coal_MW", "Coal")

    elif "gas" in query:
        return get_average_generation("Gas_MW", "Gas")

    elif "nuclear" in query:
        return get_average_generation("Nuclear_MW", "Nuclear")

    elif "total" in query or "generation" in query:
        return get_average_generation("Total_Generation_MW", "Total Energy")

    # Trend tool
    elif contains_any(query, trend_keywords):
        return (
            "Tool Selected: Trend Analysis Tool\n\n"
            "The smart grid dataset shows time-based variation in energy generation. "
            "The system can analyze changes in solar, wind, hydro, coal, gas, nuclear, "
            "renewable share, and total generation."
        )

    else:
        return (
            "I can answer smart-grid questions about solar, wind, hydro, coal, gas, "
            "nuclear, total generation, renewable share, forecasting, anomalies, trends, "
            "maximum/minimum generation dates, and comparisons."
        )

# =========================
# USER INTERFACE
# =========================

st.subheader("Ask the Smart Grid Agent")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is the date of maximum wind generation?"
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
    "Predict next energy generation",
    "What is the date of maximum solar generation?",
    "What is the date of maximum wind generation?",
    "When was hydro generation lowest?",
    "Show coal generation",
    "Analyze gas power",
    "Show nuclear generation",
    "What is the renewable share?",
    "Detect abnormal energy generation",
    "Compare solar and wind",
    "Which source generates more energy?",
    "Explain energy trend"
]

st.write(sample_questions)

# =========================
# OPTIONAL DISPLAY SECTIONS
# =========================

#st.subheader("Model Evaluation Results")
#st.dataframe(results)

#st.subheader("Dataset Preview")
#st.dataframe(df.head())

#st.subheader("Total Energy Generation Over Time")
#st.line_chart(df.set_index("Date")["Total_Generation_MW"])
