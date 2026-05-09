import json
import time

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from spark_config import create_spark_session


def train_models():

    # =====================================================
    # Start Spark
    # =====================================================

    spark = create_spark_session()

    print("\nLoading Feature Engineered Datasets...\n")

    # =====================================================
    # Load Train/Test Data
    # =====================================================

    import pandas as pd

    train_pd = pd.read_pickle(
    "data/processed/train_data.pkl"
)

    test_pd = pd.read_pickle(
    "data/processed/test_data.pkl"
    )

    train_df = spark.createDataFrame(train_pd)

    test_df = spark.createDataFrame(test_pd)

    print(f"Train Rows: {train_df.count()}")
    print(f"Test Rows: {test_df.count()}")

    # =====================================================
    # Logistic Regression
    # =====================================================

    print("\n====================================")
    print("Training Logistic Regression...")
    print("====================================\n")

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10
    )

    lr_start = time.time()

    lr_model = lr.fit(train_df)

    lr_end = time.time()

    lr_training_time = lr_end - lr_start

    # =====================================================
    # Predict
    # =====================================================

    lr_predictions = lr_model.transform(test_df)

    print("\nLogistic Regression Predictions:\n")

    lr_predictions.select(
        "label",
        "prediction",
        "probability"
    ).show(5, truncate=False)

    # =====================================================
    # Evaluate
    # =====================================================

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    lr_auc = evaluator.evaluate(lr_predictions)

    print(f"\nLogistic Regression AUC: {lr_auc:.4f}")
    
    # =====================================================
    # Save Logistic Regression Artifacts
    # =====================================================

    lr_artifacts = {

        "model_type": "Logistic Regression",

        "auc": float(lr_auc),

        "training_time": float(lr_training_time),

        "intercept": float(lr_model.intercept),

        "coefficients": lr_model.coefficients.toArray().tolist()

    }

    with open(
        "models/logistic_regression_model.json",
        "w"
    ) as f:

       json.dump(
           lr_artifacts,
           f,
           indent=4
    )

    print("\nLogistic Regression Artifacts Saved.")

    print(
        f"Logistic Regression Training Time: "
        f"{lr_training_time:.2f} seconds"
    )

    # =====================================================
    # Save Model
    # =====================================================

    import pickle

    

    # =====================================================
    # Random Forest
    # =====================================================

    print("\n====================================")
    print("Training Random Forest...")
    print("====================================\n")

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=50,
        maxDepth=10
    )

    rf_start = time.time()

    rf_model = rf.fit(train_df)

    rf_end = time.time()

    rf_training_time = rf_end - rf_start

    # =====================================================
    # Predict
    # =====================================================

    rf_predictions = rf_model.transform(test_df)

    print("\nRandom Forest Predictions:\n")

    rf_predictions.select(
        "label",
        "prediction",
        "probability"
    ).show(5, truncate=False)

    # =====================================================
    # Evaluate
    # =====================================================

    rf_auc = evaluator.evaluate(rf_predictions)

    print(f"\nRandom Forest AUC: {rf_auc:.4f}")
    
    # =====================================================
    # Save Random Forest Artifacts
    # =====================================================

    rf_artifacts = {

        "model_type": "Random Forest",

        "auc": float(rf_auc),

        "training_time": float(rf_training_time),

        "feature_importances":
        rf_model.featureImportances.toArray().tolist()

    }

    with open(
        "models/random_forest_model.json",
        "w"
    ) as f:

        json.dump(
           rf_artifacts,
           f,
           indent=4
    )

    print("\nRandom Forest Artifacts Saved.")

    print(
        f"Random Forest Training Time: "
        f"{rf_training_time:.2f} seconds"
    )

    # =====================================================
    # Save Model
    # =====================================================

    print("\nLogistic Regression Training Complete.")
    
    print("\nRandom Forest Training Complete.")

    # =====================================================
    # Final Comparison
    # =====================================================

    print("\n====================================")
    print("FINAL MODEL COMPARISON")
    print("====================================\n")

    print(f"Logistic Regression AUC : {lr_auc:.4f}")
    print(f"Random Forest AUC       : {rf_auc:.4f}")

    print(
        f"Logistic Regression Time: "
        f"{lr_training_time:.2f} sec"
    )

    print(
        f"Random Forest Time      : "
        f"{rf_training_time:.2f} sec"
    )

    # =====================================================
    # Stop Spark
    # =====================================================

    spark.stop()

    print("\nWeek 3 Complete.\n")


if __name__ == "__main__":
    train_models()