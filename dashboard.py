
import os
import sys
import glob
from pathlib import Path

# Configure the existing Windows Spark installation before importing PySpark.
SPARK_HOME = Path(r"C:\Spark")
os.environ["JAVA_HOME"] = r"C:\Java\jdk1.8.0_481"
os.environ["SPARK_HOME"] = str(SPARK_HOME)

PYTHON312 = r"C:\Users\nksir\AppData\Local\Programs\Python\Python312\python.exe"

os.environ["PYSPARK_PYTHON"] = PYTHON312
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON312

os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

for spark_path in [
    SPARK_HOME / "python",
    SPARK_HOME / "python" / "lib" / "pyspark.zip",
    SPARK_HOME / "python" / "lib" / "py4j-0.10.9.7-src.zip",
]:
    if spark_path.exists() and str(spark_path) not in sys.path:
        sys.path.insert(0, str(spark_path))

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_FILE = OUTPUT_DIR / "metrics.txt"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
COMPARISON_FILE = OUTPUT_DIR / "algorithm_comparison.csv"
MODEL_PATH = OUTPUT_DIR / "student_pass_model"

FEATURE_COLUMNS = [
    "age", "Medu", "Fedu", "studytime", "failures",
    "absences", "G1", "G2", "traveltime", "schoolsup",
    "famsup", "internet", "health",
]

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Academic Performance Prediction")
st.caption(
    "INT315 Cluster Computing | Apache Spark 3.5.9 | "
    "PySpark MLlib | UCI Student Performance Dataset"
)


# ---------------------- Load saved results ----------------------

if not METRICS_FILE.exists():
    st.error("Metrics file not found. Run model.py first.")
    st.stop()

if not COMPARISON_FILE.exists():
    st.error("Algorithm comparison not found. Run model.py first.")
    st.stop()

if not PREDICTIONS_DIR.exists():
    st.error("Prediction results not found. Run model.py first.")
    st.stop()

metrics_text = METRICS_FILE.read_text(encoding="utf-8")


def read_metric(name):
    for line in metrics_text.splitlines():
        if line.startswith(name + ":"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def read_text_metric(name):
    for line in metrics_text.splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return "Unknown"


accuracy = read_metric("Accuracy")
precision = read_metric("Weighted precision")
recall = read_metric("Weighted recall")
f1 = read_metric("F1 score")
selected_model_name = read_text_metric("Model")

comparison = pd.read_csv(COMPARISON_FILE)
comparison = comparison.sort_values(
    by=["f1_score", "accuracy"],
    ascending=False,
).reset_index(drop=True)

csv_files = glob.glob(str(PREDICTIONS_DIR / "part-*.csv"))
if not csv_files:
    st.error("Prediction CSV files are missing. Rerun model.py.")
    st.stop()

df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True,
)


# ---------------------- Model evaluation ----------------------

st.header("1. Model Evaluation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("Weighted Precision", f"{precision:.4f}")
c3.metric("Weighted Recall", f"{recall:.4f}")
c4.metric("F1 Score", f"{f1:.4f}")

st.success(f"Selected model: {selected_model_name}")
st.caption(
    "Evaluation uses the held-out test split from this experiment. "
    "Performance on future students may differ."
)


# ---------------------- Algorithm comparison ----------------------

st.header("2. Algorithm Comparison")
st.write(
    "All algorithms were trained using the same training/test split "
    "and the same feature set."
)

comparison_display = comparison.rename(
    columns={
        "algorithm": "Algorithm",
        "accuracy": "Accuracy",
        "weighted_precision": "Weighted Precision",
        "weighted_recall": "Weighted Recall",
        "f1_score": "F1 Score",
        "training_time_seconds": "Training Time (seconds)",
    }
).copy()

for metric_column in [
    "Accuracy", "Weighted Precision", "Weighted Recall", "F1 Score"
]:
    comparison_display[metric_column] = comparison_display[
        metric_column
    ].round(4)

comparison_display["Training Time (seconds)"] = (
    comparison_display["Training Time (seconds)"].round(3)
)

st.dataframe(comparison_display, use_container_width=True, hide_index=True)

st.subheader("Accuracy vs F1 Score")
chart_data = comparison.set_index("algorithm")[
    ["accuracy", "f1_score"]
].rename(
    columns={"accuracy": "Accuracy", "f1_score": "F1 Score"}
)
st.bar_chart(chart_data)

winner = comparison.iloc[0]
st.info(
    f"Best model in this run: {winner['algorithm']} "
    f"(F1 = {winner['f1_score']:.4f}, "
    f"accuracy = {winner['accuracy'] * 100:.2f}%). "
    "Selection is based on test-set F1, with accuracy as the tie-breaker."
)


# ---------------------- Existing prediction overview ----------------------

st.header("3. Prediction Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Test Students", len(df))
c2.metric(
    "Predicted PASS",
    int((df["predicted_result"] == "PASS").sum()),
)
c3.metric(
    "Predicted AT RISK",
    int((df["predicted_result"] == "AT RISK").sum()),
)

left, right = st.columns(2)

with left:
    st.subheader("Predicted Outcomes")
    outcome_counts = (
        df["predicted_result"]
        .value_counts()
        .rename_axis("Outcome")
        .to_frame("Students")
    )
    st.bar_chart(outcome_counts)

with right:
    st.subheader("Actual vs Predicted")
    matrix = pd.crosstab(
        df["actual_result"],
        df["predicted_result"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    st.dataframe(matrix, use_container_width=True)


# ---------------------- New student prediction ----------------------

st.header("4. Predict for a New Student")
st.write(
    "Enter information for a student to estimate their final pass/at-risk "
    "outcome. Do not enter the final grade: the model uses earlier grades "
    "G1 and G2 along with student attributes."
)

with st.form("new_student_form"):
    st.subheader("Academic and Personal Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age", min_value=15, max_value=22, value=16
        )
        medu = st.selectbox(
            "Mother's education (0–4)", [0, 1, 2, 3, 4], index=2
        )
        fedu = st.selectbox(
            "Father's education (0–4)", [0, 1, 2, 3, 4], index=2
        )
        studytime = st.selectbox(
            "Weekly study time (1–4)", [1, 2, 3, 4], index=1
        )
        failures = st.number_input(
            "Past class failures", min_value=0, max_value=4, value=0
        )

    with col2:
        absences = st.number_input(
            "School absences", min_value=0, max_value=100, value=2
        )
        g1 = st.number_input(
            "First-period grade G1 (0–20)",
            min_value=0, max_value=20, value=10
        )
        g2 = st.number_input(
            "Second-period grade G2 (0–20)",
            min_value=0, max_value=20, value=10
        )
        traveltime = st.selectbox(
            "Travel time to school (1–4)", [1, 2, 3, 4], index=1
        )

    with col3:
        schoolsup = st.selectbox(
            "Extra educational support", ["No", "Yes"]
        )
        famsup = st.selectbox(
            "Family educational support", ["No", "Yes"]
        )
        internet = st.selectbox(
            "Internet access at home", ["Yes", "No"]
        )
        health = st.selectbox(
            "Current health (1–5)", [1, 2, 3, 4, 5], index=2
        )

    submitted = st.form_submit_button(
        "Predict Student Outcome",
        type="primary",
    )

if submitted:
    if not MODEL_PATH.exists():
        st.error(
            "Saved model not found. Run model.py and check its output first."
        )
    else:
        try:
            with st.spinner("Loading Spark model and generating prediction..."):

                @st.cache_resource
                def get_spark_session():
                    return (
                        SparkSession.builder
                        .appName("INT315_Student_Performance_Dashboard")
                        .master("local[*]")
                        .getOrCreate()
                    )

                @st.cache_resource
                def load_model(path):
                    return PipelineModel.load(path)

                spark = get_spark_session()
                saved_model = load_model(str(MODEL_PATH))

                new_student = {
                    "age": int(age),
                    "Medu": int(medu),
                    "Fedu": int(fedu),
                    "studytime": int(studytime),
                    "failures": int(failures),
                    "absences": int(absences),
                    "G1": int(g1),
                    "G2": int(g2),
                    "traveltime": int(traveltime),
                    "schoolsup": 1.0 if schoolsup == "Yes" else 0.0,
                    "famsup": 1.0 if famsup == "Yes" else 0.0,
                    "internet": 1.0 if internet == "Yes" else 0.0,
                    "health": int(health),
                }

                input_df = spark.createDataFrame([new_student])
                prediction = saved_model.transform(input_df).first()

                predicted_class = int(prediction["prediction"])
                predicted_label = (
                    "PASS" if predicted_class == 1 else "AT RISK"
                )

            st.subheader("Prediction Result")

            if predicted_class == 1:
                st.success("Model prediction: LIKELY TO PASS")
            else:
                st.warning("Model prediction: AT RISK OF FAILING")

            st.write(f"Predicted class: {predicted_class}")
            st.caption(
                "This is a model estimate, not a guaranteed outcome or "
                "a definitive judgment about the student."
            )

            with st.expander("Review submitted inputs"):
                st.json(new_student)

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info(
                "Check that the saved Spark model exists and that the "
                "Java and Spark paths are configured correctly."
            )


# ---------------------- Student records ----------------------

st.header("5. Test-Set Student Records")

choices = ["All"] + sorted(
    df["predicted_result"].dropna().unique().tolist()
)
selected_result = st.selectbox(
    "Filter records by predicted outcome",
    choices,
)

filtered = df.copy()
if selected_result != "All":
    filtered = filtered[
        filtered["predicted_result"] == selected_result
    ]

display_columns = [
    "age", "studytime", "absences", "G1", "G2", "G3",
    "actual_result", "predicted_result",
]

st.caption(f"Showing {len(filtered)} of {len(df)} test records.")
st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered[display_columns].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download displayed predictions (CSV)",
    data=csv_data,
    file_name="student_predictions.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Dataset: UCI Student Performance — Mathematics. "
    "Passing is defined as G3 >= 10. "
    "Predictions are estimates, not definitive judgments about students."
)
