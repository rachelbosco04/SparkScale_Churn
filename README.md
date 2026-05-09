# SparkScale Churn  
### Distributed Telecom Customer Churn Prediction Platform

SparkScale Churn is a production-style machine learning platform designed to predict telecom customer churn using distributed data processing and scalable ML pipelines with Apache Spark.

The project demonstrates an end-to-end workflow for:
- large-scale feature engineering
- distributed model training
- churn probability prediction
- customer risk analytics
- deployment-style dashboard visualization

Built using PySpark MLlib, Dockerized Spark clusters, and an interactive frontend dashboard, the system simulates a real-world enterprise churn prediction environment.

---

# Key Features

- Distributed data processing using Apache Spark
- Feature engineering with Spark SQL
- MLlib-based Logistic Regression and Random Forest models
- Batch churn prediction pipeline
- Real-time customer churn prediction interface
- Interactive analytics dashboard
- Dockerized multi-worker Spark cluster
- Production-style deployment simulation

---

# Technology Stack

| Category | Technologies |
|---|---|
| Big Data | Apache Spark |
| Machine Learning | PySpark MLlib |
| Backend | Python |
| Cluster Environment | Docker |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Chart.js |
| Dataset | IBM Telco Customer Churn Dataset |

---

# System Architecture

```text
Raw Telecom Data
        ↓
Data Cleaning Pipeline
        ↓
Feature Engineering (Spark SQL)
        ↓
Distributed ML Training
        ↓
Model Evaluation
        ↓
Batch Prediction Pipeline
        ↓
Interactive Dashboard