# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops a **Credit Risk Probability Model** for Bati Bank’s Buy-Now-Pay-Later service, leveraging alternative data (transactional and behavioral) to predict customer creditworthiness. The model outputs risk probabilities, credit scores, and recommends optimal loan amounts and durations. The solution is built with MLOps best practices, including experiment tracking, containerized deployment, and CI/CD.

---

## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord requires robust, transparent, and auditable credit risk models to ensure regulatory compliance and adequate capital reserves. Interpretable models (e.g., Logistic Regression with WoE) allow clear explanation of risk factors to regulators and stakeholders, while thorough documentation ensures traceability and reduces regulatory risk.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Without a direct default label, a proxy variable (using RFM metrics and clustering) is needed to approximate credit risk. This introduces risks: the proxy may misclassify customers, leading to false positives (creditworthy customers denied) or false negatives (risky customers approved), which can impact business outcomes and reputation.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Simple models are easier to interpret and audit, making them regulator-friendly, but may miss complex patterns. Complex models (e.g., Gradient Boosting) can improve accuracy but are harder to interpret, requiring additional explainability tools to satisfy regulatory requirements.

---

## Project Structure

```
credit-risk-scoring-model/
├── data/                  # Raw and processed data
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks for EDA, feature engineering, modeling
├── plots/                 # Visualizations
├── src/
│   ├── api/               # FastAPI app and Pydantic models
│   ├── data_processing.py # Data processing and feature engineering pipeline
│   └── train.py           # Model training and MLflow tracking
├── tests/                 # Unit tests
├── Dockerfile             # Containerization
├── docker-compose.yml     # Container orchestration
├── requirements.txt       # Python dependencies
└── .github/workflows/ci.yml # CI/CD pipeline
```

---

## Workflow

### 1. Data Processing & Feature Engineering

- All logic is in `src/data_processing.py` using `sklearn.pipeline.Pipeline`.
- Features include:
  - Aggregate: Recency, Frequency, Monetary, Avg/Std Transaction Amount
  - Extracted: Transaction Hour, Day, Month, Year
  - Categorical encoding (LabelEncoder)
  - Missing value imputation (median)
  - Normalization/standardization (StandardScaler)
- Proxy target (`is_high_risk`) is created using RFM metrics and KMeans clustering.

### 2. Model Training & Tracking

- Implemented in `src/train.py`.
- Trains Logistic Regression and Gradient Boosting models.
- Hyperparameter tuning with GridSearchCV.
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- MLflow used for experiment tracking.
- Best model saved to `models/best_model.pkl`.

### 3. API Deployment

- FastAPI app in `src/api/main.py`.
- `/predict` endpoint accepts customer features and returns risk probability and high-risk flag.
- Pydantic models in `src/api/pydantic_models.py` for request/response validation.
- Containerized with Docker and orchestrated with Docker Compose.

### 4. CI/CD

- GitHub Actions workflow in `.github/workflows/ci.yml`.
- Runs `flake8` for linting and `pytest` for unit tests on every push or PR.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ruthina-27/credit-risk-scoring-model.git
cd credit-risk-scoring-model
```

### 2. Set Up Environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Data Processing & Model Training

```bash
python src/data_processing.py
python src/train.py
```

### 4. Run the API Locally

```bash
uvicorn src.api.main:app --reload
# or with Docker Compose
docker-compose up --build
```

### 5. Make Predictions

Send a POST request to `http://localhost:8000/predict` with a JSON body like:

```json
{
  "Recency": 10,
  "Frequency": 5,
  "Monetary": 1000,
  "AvgTransactionAmount": 200,
  "StdTransactionAmount": 50,
  "MostCommonHour": 14,
  "MostCommonDay": 3,
  "MostCommonMonth": 6,
  "MostCommonYear": 2024
}
```

Response:

```json
{
  "risk_probability": 0.23,
  "is_high_risk": 0
}
```

---

## Testing & CI

- Run all unit tests:
  ```bash
  pytest tests/
  ```
- Lint code:
  ```bash
  flake8 src/ tests/
  ```
- CI/CD runs automatically on GitHub via `.github/workflows/ci.yml`.

---

## References

- [Basel II Capital Accord](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [How to Develop a Credit Risk Model and Scorecard](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [CFI: Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Risk Officer: Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)
