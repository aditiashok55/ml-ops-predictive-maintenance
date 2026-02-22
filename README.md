# Production-Grade ML Platform — Predictive Maintenance

An end-to-end MLOps platform that predicts **Remaining Useful Life (RUL)** of
industrial jet engines from time-series sensor data. RUL represents the number
of operational cycles remaining before an engine requires maintenance or risks
failure — enabling maintenance teams to intervene proactively rather than
reactively. Early and accurate RUL prediction directly reduces unplanned
downtime, lowers maintenance costs, and improves operational safety.

The platform is built on the NASA CMAPSS dataset, which simulates real-world
turbofan engine degradation across 21 sensors over hundreds of operational
cycles. The ML system ingests these sensor readings, extracts degradation
signals via rolling window feature engineering, and serves real-time
predictions through a production-grade REST API.

Built to demonstrate end-to-end ML engineering practices including experiment
tracking, containerized inference, CI/CD automation, and real-time monitoring
— mirroring workflows used in industrial ML deployments across aerospace,
automotive, and manufacturing sectors.

---

## Problem Statement

Unplanned equipment failure costs industrial operators billions annually.
This platform ingests time-series sensor readings from jet engines and predicts
how many operational cycles remain before failure — enabling maintenance teams
to act before breakdowns occur.

Dataset: [NASA CMAPSS Turbofan Engine Degradation](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Architecture
```
Sensor Data → Preprocessing → Training Pipeline → MLflow Registry
                                                          ↓
                                                 FastAPI Inference Service
                                                          ↓
                                              Prometheus → Grafana Dashboard

CI/CD: GitHub Actions → Docker Hub
Infra:  Terraform → AWS (EC2 + S3 + VPC + IAM)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Model Training | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow |
| Inference API | FastAPI, Pydantic, Uvicorn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |
| Infrastructure | Terraform, AWS (EC2, S3, IAM, VPC) |

---

## Project Structure
```
ml-platform/
├── src/
│   ├── data/
│   │   └── preprocess.py       # Data ingestion and feature engineering
│   ├── training/
│   │   └── train.py            # XGBoost training + MLflow logging
│   └── inference/
│       └── app.py              # FastAPI inference service
├── tests/                      # Unit tests for all components
├── docker/
│   └── Dockerfile.inference    # Inference service container
├── terraform/                  # AWS infrastructure as code
├── .github/workflows/
│   └── ci.yml                  # CI/CD pipeline
├── docker-compose.yml          # Full local stack
└── prometheus.yml              # Metrics scraping config
```

---

## ML Pipeline

**Preprocessing** — Drops low-variance sensors, computes rolling window
statistics (mean, std over 5 cycles) per engine, normalizes sensor readings.

**Training** — XGBoost regressor trained to predict RUL. Experiments tracked
in MLflow with full parameter, metric, and artifact logging. 

**Model Registry** — Trained models registered and versioned in MLflow.
Inference service loads by registry name, enabling zero-downtime model updates.

---

## Model Performance

| Metric | Train | Validation |
|---|---|---|
| RMSE | 28.24 | 36.16 |
| MAE | — | 25.51 |
| R² | — | 0.714 |

Baseline XGBoost with rolling window feature engineering.
Mild overfitting (train/val RMSE gap) identified — regularization
tuning is tracked as a future experiment in MLflow.

---

## Running Locally

**Prerequisites:** Docker Desktop, Python 3.12

**Clone and start the full stack:**
```bash
git clone https://github.com/your-username/ml-ops-predictive-maintenance
cd ml-ops-predictive-maintenance
docker compose up
```

**Services:**
| Service | URL |
|---|---|
| Inference API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

**Example prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1, "cycle": 50,
    "op_setting_1": 0.5, "op_setting_2": 0.4, "op_setting_3": 0.3,
    "sensor_2": 0.6, "sensor_3": 0.5, "sensor_4": 0.7,
    "sensor_7": 0.4, "sensor_8": 0.8, "sensor_9": 0.6,
    "sensor_11": 0.5, "sensor_12": 0.4, "sensor_13": 0.6,
    "sensor_14": 0.7, "sensor_15": 0.3, "sensor_17": 0.5,
    "sensor_20": 0.4, "sensor_21": 0.6
  }'
```

**Response:**
```json
{
  "engine_id": 1,
  "cycle": 50,
  "predicted_rul": 57.19,
  "warning_level": "WARNING",
  "latency_ms": 130.1
}
```

---

## CI/CD Pipeline

Every push to `main` triggers:
1. Runs full test suite (7 tests across data, training, inference layers)
2. Builds Docker image
3. Pushes to Docker Hub with `latest` and commit SHA tags

---

## Infrastructure (Terraform)

AWS infrastructure defined as code in `terraform/`. Provisions:
- VPC with public subnet and internet gateway
- EC2 t2.micro instance (free tier) running the inference container
- S3 bucket with versioning and encryption for model artifacts
- IAM role with least-privilege S3 access
- Security groups for API, MLflow, and Grafana ports
```bash
cd terraform
terraform init
terraform plan   # Preview infrastructure
terraform apply  # Deploy to AWS
```

---

## Monitoring

Prometheus scrapes metrics from the inference service every 15 seconds.
Grafana dashboard tracks request rate, p95 latency, and prediction
distribution in real time.

---

## Tests
```bash
pytest tests/ -v
```

7 tests covering preprocessing correctness, feature engineering, model
evaluation metrics, and API endpoint behavior.

---

## Future Work

- Hyperparameter tuning with MLflow experiment comparison
- Drift detection using Evidently AI when sensor distributions shift
- Multi-dataset training (FD002–FD004) for generalization
- Kubernetes deployment for horizontal scaling
- AWS deployment via existing Terraform configuration