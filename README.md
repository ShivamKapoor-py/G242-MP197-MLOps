# G242 – MP197: Metro Station Crowd Prediction (Kochi)

## Group Information

- **Group:** G242
- **Problem ID:** MP197 – Metro Station Crowd
- **Batch:** 54
- **Faculty:** Dr. Upasna Joshi

### Team Members

| Student ID | Name |
|---|---|
| S24CSEU2496 | Aarav Chaudhary |
| S24CSEU0044 | Krishna Solanki |
| S24CSEU0414 | Shivam Kapoor |
| S24CSEU1188 | Hardik Garg |

---

# 1. Project Overview

The **MP197 – Metro Station Crowd Prediction** project focuses on predicting hourly passenger entries at metro stations in Kochi using machine learning.

The project follows an end-to-end MLOps workflow that includes dataset auditing, data cleaning, preprocessing, model training, evaluation, experiment tracking, data drift monitoring, retraining decisions, model validation, and deployment preparation.

The objective is not only to build an accurate machine learning model, but also to establish a reproducible and maintainable workflow for monitoring and improving the model over time.

---

# 2. Project Objectives

- Audit and understand the provided datasets.
- Identify data quality issues such as duplicate records, sensor anomalies, and inconsistent categories.
- Clean and preprocess the dataset without modifying the original raw data.
- Train and compare multiple machine learning regression models.
- Track experiments, parameters, metrics, and model versions using MLflow.
- Evaluate model performance on later data.
- Monitor feature drift and identify changes in the data distribution.
- Determine when model retraining is required.
- Train and validate candidate models against the existing champion model.
- Serve predictions through a REST API.
- Prepare the application for containerization and deployment.

---

# 3. Project Workflow

The project follows the workflow below:

1. Dataset auditing
2. Data cleaning and quality checks
3. Feature preprocessing
4. Model training and comparison
5. Experiment tracking using MLflow
6. Model selection and registration
7. Evaluation on later data
8. Data drift analysis
9. Retraining decision
10. Candidate model training and validation
11. API-based model serving
12. Docker containerization
13. Kubernetes deployment configuration

---

# 4. Repository Structure

```text
G242_MP197/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── later.csv
│   │   └── holdout_features.csv
│   │
│   └── processed/
│       ├── train_clean.csv
│       └── cleaning_report.json
│
├── src/
│   ├── data_audit.py
│   │
│   └── mp197/
│       ├── data_cleaning.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── drift.py
│       ├── retraining_decision.py
│       ├── retrain_candidate.py
│       └── api.py
│
├── models/
│   └── champion/
│
├── reports/
│   ├── findings.yaml
│   └── metrics/
│       ├── training_results.json
│       ├── later_evaluation.json
│       ├── drift_report.json
│       ├── retraining_decision.json
│       ├── candidate_validation.json
│       └── later_predictions.csv
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 5. Stage 1 – Dataset Setup and Audit

The original datasets must be preserved without modification.

## Important Rules

- Keep the original CSV files unchanged.
- Do not open and resave `train.csv` in Excel.
- Perform all cleaning operations through scripts.
- Maintain reproducibility and traceability of the data.

## Expected Initial Dataset Layout

```text
G242_MP197/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── later.csv
│       └── holdout_features.csv
├── src/
│   └── data_audit.py
├── reports/
│   └── findings.yaml
├── README.md
└── .gitignore
```

## Environment Setup

Create and activate a Python virtual environment.

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the initial Stage 1 dependencies:

```bash
pip install pandas pyyaml dvc
```

Initialize Git and DVC:

```bash
git init
dvc init
```

## Dataset Tracking

Track the datasets using DVC:

```bash
dvc add data/raw/train.csv data/raw/later.csv
```

Commit the project initialization:

```bash
git add .
git commit -m "Initialize project and track raw datasets"
```

## Run the Dataset Audit

```bash
python src/data_audit.py \
    --train data/raw/train.csv \
    --later data/raw/later.csv \
    --holdout data/raw/holdout_features.csv
```

The audit script is read-only and checks:

- Dataset structure and columns
- Missing values
- Duplicate records
- Basic data quality
- Dataset consistency

Review the audit output and update the Stage 1 findings in:

```text
reports/findings.yaml
```

---

# 6. Data Cleaning and Preprocessing

The data cleaning pipeline addresses the data quality issues identified during dataset auditing.

The cleaning process includes:

- Removing duplicate rows
- Normalizing category spelling and naming
- Correcting identified temperature sensor anomalies
- Validating the cleaned dataset
- Checking for missing values
- Ensuring expected categorical values are present

The original raw data is retained, while cleaned data is saved separately in:

```text
data/processed/
```

Preprocessing is handled through a reusable pipeline that:

- Encodes categorical features using One-Hot Encoding
- Passes numerical features through the preprocessing pipeline
- Ensures consistent transformations during training and inference

---

# 7. Machine Learning Model Training

The project compares multiple regression models to predict hourly metro entries.

The models evaluated are:

- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

The dataset is divided into training and testing subsets, and the models are evaluated using:

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R² Score** – Coefficient of determination

The model with the lowest MAE is selected as the best-performing model for the initial training stage.

---

# 8. Experiment Tracking and Model Management

MLflow is used for experiment tracking and model management.

It records:

- Model parameters
- Evaluation metrics
- Experiment runs
- Model artifacts
- Model versions
- Dataset hash for reproducibility

The project uses a registered model named:

```text
MP197MetroCrowdRegressor
```

The selected model is managed using the `champion` alias, allowing the serving application to load the designated model.

The local MLflow tracking database is:

```text
mlflow.db
```

---

# 9. Model Evaluation on Later Data

The selected champion model is evaluated on the later dataset to assess how well it performs on newer observations.

The evaluation includes:

- MAE
- RMSE
- R² Score
- Row-level actual and predicted values
- Absolute prediction errors

The results are saved under:

```text
reports/metrics/
```

This evaluation helps identify whether the model's performance remains reliable over time.

---

# 10. Data Drift Monitoring

Data drift monitoring compares the feature distributions of the training dataset with those of the later dataset.

The project uses:

- **Population Stability Index (PSI)** for measuring distribution changes
- **Kolmogorov–Smirnov (KS) test** for statistical comparison of numerical features

The drift analysis classifies PSI values as:

| PSI Value | Interpretation |
|---|---|
| Below 0.10 | Low drift |
| 0.10 to below 0.20 | Moderate drift |
| 0.20 or higher | High drift |

The drift report identifies features whose distributions have changed and saves the results to:

```text
reports/metrics/drift_report.json
```

---

# 11. Retraining Decision

The retraining decision module combines model performance degradation and feature drift.

The system recommends retraining when both of the following conditions are met:

1. The later-dataset MAE is at least 1.5 times the original test MAE.
2. At least one feature exhibits high drift.

The decision is recorded in:

```text
reports/metrics/retraining_decision.json
```

The system can produce two outcomes:

- `RETRAINING_RECOMMENDED`
- `CONTINUE_MONITORING`

A retraining recommendation does not automatically promote a new model. A candidate must first be trained and validated.

---

# 12. Candidate Model Training and Validation

When retraining is recommended, a new candidate model is trained using the available updated data.

The candidate is evaluated against the existing champion using a validation dataset.

The comparison includes:

- MAE
- RMSE
- R²
- Percentage improvement in MAE

The candidate model is registered in MLflow with a new version.

The existing champion remains unchanged unless the candidate is explicitly approved for promotion.

This prevents an unvalidated model from automatically replacing the production model.

---

# 13. Model Serving API

The project uses FastAPI to serve predictions through a REST API.

## Endpoints

### Health Check

```http
GET /health
```

Returns the health status of the API.

### Prediction

```http
POST /predict
```

Accepts metro-related input features and returns the predicted hourly entries.

The API validates inputs using Pydantic, rejects invalid values, and logs prediction events in structured JSON format.

The API also provides automatic interactive documentation through Swagger UI.

---

# 14. Docker Containerization

The project includes a Dockerfile to package the application and its dependencies.

Docker is used to:

- Create a consistent runtime environment
- Package the API and model
- Reduce environment-related deployment problems
- Simplify application deployment

Build the Docker image:

```bash
docker build -t mp197-api:1.0 .
```

Run the container:

```bash
docker run -p 8000:8000 mp197-api:1.0
```

The API can then be accessed locally at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 15. Kubernetes Deployment

The project includes Kubernetes configuration files for deployment.

The deployment configuration includes:

- Multiple application replicas
- Startup probes
- Readiness probes
- Liveness probes
- CPU and memory requests
- CPU and memory limits

The Kubernetes Service exposes the API within the cluster and forwards requests to the application container.

Configuration files are available under:

```text
k8s/
```

---

# 16. Testing

Automated tests are included to verify important pipeline and API functionality.

The tests cover:

- Data cleaning
- Duplicate removal
- Category normalization
- Preprocessing transformations
- Model prediction pipeline
- API health endpoint
- Valid prediction requests
- Invalid input validation
- Rejection of unexpected input fields

Run tests using:

```bash
pytest
```

---

# 17. Project Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data handling and cleaning |
| NumPy | Numerical operations |
| Scikit-learn | ML models and preprocessing |
| MLflow | Experiment tracking and model management |
| FastAPI | REST API |
| Pydantic | Input validation |
| Uvicorn | API server |
| DVC | Dataset versioning |
| Git | Source control |
| Docker | Containerization |
| Kubernetes | Deployment orchestration |
| Pytest | Automated testing |

---

# 18. Important Project Guidelines

- Keep the original dataset files unchanged.
- Do not share the group's datasets or project code with other groups.
- Maintain reproducibility wherever possible.
- Track meaningful changes through Git commits.
- Every team member should contribute through their own GitHub account.
- All four team members should be added as repository collaborators.
- Every team member must understand and be able to explain the code during the viva.

---

# 19. Suggested Git Commit History

Use meaningful commits to document the development process.

```bash
git add README.md .gitignore
git commit -m "Add project documentation and ignore files"

git add src/data_audit.py
git commit -m "Add dataset audit script"

git add data/raw/*.dvc
git commit -m "Track datasets with DVC"

git add reports/findings.yaml
git commit -m "Add Stage 1 dataset audit findings"

git add src/mp197/data_cleaning.py
git commit -m "Implement data cleaning pipeline"

git add src/mp197/preprocessing.py
git commit -m "Add feature preprocessing pipeline"

git add src/mp197/train.py
git commit -m "Train and compare regression models"

git add src/mp197/evaluate.py
git commit -m "Add temporal model evaluation"

git add src/mp197/drift.py
git commit -m "Implement feature drift monitoring"

git add src/mp197/retraining_decision.py
git commit -m "Add conditional retraining decision logic"

git add src/mp197/retrain_candidate.py
git commit -m "Train and validate candidate model"

git add src/mp197/api.py
git commit -m "Implement prediction API"

git add Dockerfile k8s/
git commit -m "Add containerization and Kubernetes deployment configuration"

git add tests/
git commit -m "Add pipeline and API tests"
```

---

# 20. Conclusion

The G242 MP197 Metro Station Crowd Prediction project demonstrates an end-to-end machine learning and MLOps workflow.

It covers the complete lifecycle of a machine learning system, from dataset auditing and model development to monitoring, retraining decisions, model validation, API serving, containerization, and deployment preparation.

The project emphasizes reproducibility, model reliability, monitoring, and controlled model updates rather than treating machine learning as a one-time training task.
