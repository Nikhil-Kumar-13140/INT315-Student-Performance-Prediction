# Student Academic Performance Prediction Using PySpark MLlib

An end-to-end machine learning project that predicts student academic outcomes using Apache Spark MLlib. The project compares four classification algorithms and provides an interactive Streamlit dashboard for exploring results and predicting outcomes for new students.

## Project Overview

The objective is to analyze student-related attributes and previous grades to predict whether a student is likely to pass or be at risk of failing.

The project demonstrates distributed data processing concepts, machine learning pipelines, model evaluation, and interactive data visualization.

## Key Features

- Data loading and preprocessing using PySpark.
- Student pass/fail classification using the final grade.
- Comparison of four machine learning algorithms.
- Evaluation using accuracy, precision, recall, F1-score, and training time.
- Best-model selection based on test F1-score, with accuracy as a tie-breaker.
- Interactive Streamlit dashboard.
- New-student prediction form.
- Student prediction results and downloadable CSV data.
- Automated launch scripts for model training and dashboard startup.

## Technologies Used

- **Language:** Python
- **Big Data Framework:** Apache Spark
- **Machine Learning:** PySpark MLlib
- **Web Dashboard:** Streamlit
- **Data Processing:** Pandas, PySpark DataFrames
- **Visualization:** Streamlit charts
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

## Dataset

**Dataset:** Student Performance Dataset — Mathematics (`student-mat.csv`)

**Source:** UCI Machine Learning Repository

Dataset link: https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains 395 student records and 33 attributes, including demographic information, family and school-related factors, study time, absences, and grades.

The model uses selected student attributes and previous grades (`G1` and `G2`) to predict the final pass/fail outcome derived from `G3`.

- **Pass:** Final grade (`G3`) is greater than or equal to 10.
- **At risk / Fail:** Final grade (`G3`) is below 10.

The prediction is a classification result, not a guarantee of a student's future academic performance.

## Machine Learning Algorithms

The project compares the following algorithms:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Logistic Regression
4. Gradient-Boosted Trees Classifier

### Previously observed results

| Algorithm | Accuracy | F1-score |
|---|---:|---:|
| Decision Tree | 85.96% | 85.66% |
| Random Forest | 92.98% | 92.98% |
| Logistic Regression | 91.23% | 91.14% |
| Gradient-Boosted Trees | 91.23% | 91.31% |

These results came from a single 80:20 train-test split with seed 42. The test set contained 57 records. Results may vary if the data, preprocessing, or model settings change.

Random Forest was selected in the observed run. Its score should not be treated as a guaranteed performance level on unseen real-world students.

## Project Structure

```text
INT315_Student_Performance/
├── data/
│   └── student-mat.csv
├── outputs/
│   ├── algorithm_comparison.csv
│   ├── metrics.txt
│   ├── student_pass_model/    # Generated model
│   └── predictions/           # Generated predictions
├── screenshots/
│   ├── dashboard_overview.png
│   ├── model_results.png
│   ├── student_prediction.png
│   ├── student_records.png
│   └── terminal_results.png
├── model.py
├── dashboard.py
├── run_model.ps1
├── run_dashboard.bat
├── .gitignore
└── README.md
```

The screenshot filenames above are the recommended names. Add your actual screenshots to the `screenshots/` folder.

## Prerequisites

The launch scripts are configured for the original Windows development environment. Before running the project, install or configure:

- Windows
- Python 3.12
- Java 8
- Apache Spark 3.5.9
- Streamlit and the required Python packages

The configured paths are specific to the original development computer. Update the scripts if your installation directories differ.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/INT315-Student-Performance-Prediction.git
cd INT315-Student-Performance-Prediction
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Train and evaluate the models

Open PowerShell in the project directory and run:

```powershell
.\run_model.ps1
```

This trains and evaluates the four classifiers, selects the best model, and generates the model outputs required by the dashboard.

If PowerShell blocks script execution, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_model.ps1
```

### 3. Launch the dashboard

Run:

```powershell
.\run_dashboard.bat
```

Open the local URL printed in the terminal, usually:

http://localhost:8501

Train the models before using the new-student prediction form on a fresh clone, because the generated model directory is not included in the repository.

## Screenshots

### Dashboard Overview

![Dashboard Overview](screenshots/dashboard_overview.png)

### Algorithm Comparison

![Algorithm Comparison](screenshots/model_results.png)

### New Student Prediction

![Student Prediction](screenshots/student_prediction.png)

### Student Records

![Student Records](screenshots/student_records.png)

### Model Training Results

![Model Training Results](screenshots/terminal_results.png)

## Methodology

1. Load the student dataset.
2. Prepare the target label from the final grade.
3. Select relevant input features and encode yes/no fields.
4. Split the data into training and testing sets.
5. Train four classification algorithms using Spark ML pipelines.
6. Evaluate their classification metrics and training time.
7. Select a model based on test F1-score and accuracy.
8. Save the selected pipeline and prediction outputs.
9. Display model comparisons and student predictions in Streamlit.

## Limitations

- The dataset is relatively small and may not represent all student populations.
- The evaluation uses a single random train-test split.
- Earlier grades (`G1` and `G2`) are important predictors of the final grade.
- Model performance on other datasets may differ.
- The application is an academic demonstration, not a substitute for professional educational assessment.

## Learning Outcomes

This project demonstrates practical experience with:

- Apache Spark and PySpark DataFrames
- Spark MLlib classification pipelines
- Data preprocessing and feature engineering
- Model comparison and evaluation
- Interactive dashboard development
- Git, GitHub, and reproducible project workflows

## Academic Context

**Course:** INT315 — Cluster Computing

**Project:** Student Academic Performance Prediction Using PySpark MLlib

**Purpose:** Academic coursework and demonstration of big-data machine learning concepts.
