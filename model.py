
import os
import csv
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    LogisticRegression,
    GBTClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ==========================================================
# INT315: Student Academic Performance Prediction
# Apache Spark 3.5.9 | PySpark MLlib
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions"
COMPARISON_PATH = OUTPUT_DIR / "algorithm_comparison.csv"
MODEL_PATH = OUTPUT_DIR / "student_pass_model"
METRICS_PATH = OUTPUT_DIR / "metrics.txt"

spark = (
    SparkSession.builder
    .appName("INT315_Student_Performance_Prediction")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n" + "=" * 65)
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
    print("Algorithm Comparison | Apache Spark MLlib")
    print("=" * 65)

    # 1. Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .csv(str(DATA_PATH))
    )

    total_records = df.count()
    print("\n1. DATASET INFORMATION")
    print("-" * 45)
    print("Dataset: UCI Student Performance - Mathematics")
    print("Total records:", total_records)
    print("Total columns:", len(df.columns))

    # 2. Prepare target: G3 >= 10 = PASS, otherwise AT RISK
    df = df.withColumn(
        "label",
        when(col("G3") >= 10, 1.0).otherwise(0.0)
    )

    feature_columns = [
        "age",
        "Medu",
        "Fedu",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "traveltime",
        "schoolsup",
        "famsup",
        "internet",
        "health",
    ]

    # Convert yes/no fields into numeric features.
    for column_name in ["schoolsup", "famsup", "internet"]:
        df = df.withColumn(
            column_name,
            when(col(column_name) == "yes", 1.0).otherwise(0.0)
        )

    df = df.select(*(feature_columns + ["G3", "label"])).dropna()

    print("\n2. DATA PREPARATION")
    print("-" * 45)
    print("Records after cleaning:", df.count())
    print("Class distribution:")
    df.groupBy("label").count().orderBy("label").show()

    # 3. Split once so every algorithm gets the same records.
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train_count = train.count()
    test_count = test.count()

    print("\n3. TRAIN/TEST SPLIT")
    print("-" * 45)
    print("Training records:", train_count)
    print("Testing records:", test_count)

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="error",
    )

    # 4. Algorithms to compare
    algorithms = [
        (
            "Decision Tree",
            DecisionTreeClassifier(
                labelCol="label",
                featuresCol="features",
                maxDepth=5,
                seed=42,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                numTrees=50,
                maxDepth=5,
                seed=42,
            ),
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                labelCol="label",
                featuresCol="features",
                maxIter=100,
                regParam=0.0,
            ),
        ),
        (
            "Gradient-Boosted Trees",
            GBTClassifier(
                labelCol="label",
                featuresCol="features",
                maxIter=30,
                maxDepth=3,
                seed=42,
            ),
        ),
    ]

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
    )

    def evaluate_metric(predictions, metric_name):
        return evaluator.evaluate(
            predictions,
            {evaluator.metricName: metric_name},
        )

    # 5. Train and evaluate every algorithm
    comparison_rows = []
    trained_models = {}

    print("\n4. ALGORITHM TRAINING AND EVALUATION")
    print("-" * 65)

    for algorithm_name, classifier in algorithms:
        print(f"\nTraining: {algorithm_name} ...")

        pipeline = Pipeline(stages=[assembler, classifier])

        start_time = time.perf_counter()
        fitted_pipeline = pipeline.fit(train)
        training_seconds = time.perf_counter() - start_time

        predictions = fitted_pipeline.transform(test)

        accuracy = evaluate_metric(predictions, "accuracy")
        precision = evaluate_metric(predictions, "weightedPrecision")
        recall = evaluate_metric(predictions, "weightedRecall")
        f1 = evaluate_metric(predictions, "f1")

        row = {
            "algorithm": algorithm_name,
            "accuracy": accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "f1_score": f1,
            "training_time_seconds": training_seconds,
        }

        comparison_rows.append(row)
        trained_models[algorithm_name] = fitted_pipeline

        print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 score:  {f1:.4f}")
        print(f"  Train time: {training_seconds:.3f} seconds")

    # 6. Select the best measured test F1; accuracy breaks ties.
    comparison_rows.sort(
        key=lambda row: (row["f1_score"], row["accuracy"]),
        reverse=True,
    )

    best_row = comparison_rows[0]
    best_name = best_row["algorithm"]
    best_model = trained_models[best_name]

    print("\n5. ALGORITHM COMPARISON")
    print("-" * 90)
    print(
        f"{'Algorithm':<25} {'Accuracy':>10} {'Precision':>11} "
        f"{'Recall':>10} {'F1':>10} {'Train(s)':>10}"
    )

    for row in comparison_rows:
        print(
            f"{row['algorithm']:<25} "
            f"{row['accuracy']:>10.4f} "
            f"{row['weighted_precision']:>11.4f} "
            f"{row['weighted_recall']:>10.4f} "
            f"{row['f1_score']:>10.4f} "
            f"{row['training_time_seconds']:>10.3f}"
        )

    print(f"\nBest model by test F1 score: {best_name}")

    # 7. Save comparison results as a regular CSV file.
    with open(COMPARISON_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "accuracy",
                "weighted_precision",
                "weighted_recall",
                "f1_score",
                "training_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    # 8. Save the selected model, including feature assembly.
    best_model.write().overwrite().save(str(MODEL_PATH))

    # 9. Generate dashboard predictions from the selected model.
    predictions = best_model.transform(test)

    print("\n6. SELECTED MODEL CONFUSION MATRIX")
    print("-" * 45)
    print("Rows = actual class; columns = predicted class")
    print("Class 0 = AT RISK; Class 1 = PASS")

    (
        predictions
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .show()
    )

    results = (
        predictions
        .withColumn(
            "actual_result",
            when(col("label") == 1.0, "PASS").otherwise("AT RISK"),
        )
        .withColumn(
            "predicted_result",
            when(col("prediction") == 1.0, "PASS").otherwise("AT RISK"),
        )
        .select(
            "age", "studytime", "absences", "G1", "G2", "G3",
            "actual_result", "predicted_result", "label", "prediction",
        )
    )

    (
        results.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(str(PREDICTIONS_PATH))
    )

    # 10. Save selected-model metrics in the existing dashboard format.
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        file.write("STUDENT ACADEMIC PERFORMANCE PREDICTION\n")
        file.write("Dataset: UCI Student Performance - Mathematics\n")
        file.write("Target: G3 >= 10 means PASS; otherwise AT RISK\n")
        file.write(f"Model: PySpark MLlib {best_name}\n")
        file.write(f"Dataset records after cleaning: {df.count()}\n")
        file.write(f"Training records: {train_count}\n")
        file.write(f"Testing records: {test_count}\n")
        file.write(f"Accuracy: {best_row['accuracy']:.4f}\n")
        file.write(
            f"Weighted precision: {best_row['weighted_precision']:.4f}\n"
        )
        file.write(f"Weighted recall: {best_row['weighted_recall']:.4f}\n")
        file.write(f"F1 score: {best_row['f1_score']:.4f}\n")
        file.write(f"Training time seconds: {best_row['training_time_seconds']:.4f}\n")
        file.write(f"Best model by test F1: {best_name}\n")

    print("\n7. SAMPLE PREDICTIONS")
    print("-" * 45)
    results.show(10, truncate=False)

    print("\nSUCCESS: All algorithms trained and evaluated.")
    print("Comparison CSV:", COMPARISON_PATH)
    print("Selected model:", best_name)
    print("Saved model:", MODEL_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("Metrics:", METRICS_PATH)

finally:
    spark.stop()
