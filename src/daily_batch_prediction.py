import json
import pandas as pd
from pyspark.sql.functions import col

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

from spark_config import create_spark_session


def daily_batch_prediction():

    # =====================================================
    # Start Spark
    # =====================================================

    spark = create_spark_session()

    print("\nStarting Daily Batch Prediction Job...\n")

    # =====================================================
    # Load New Monthly Customer Data
    # =====================================================

    df = spark.read.csv(
    "data/raw/telco_churn.csv",
        header=True,
        inferSchema=True
    )

    print(f"Rows Loaded: {df.count()}")

    # =====================================================
    # Basic Cleaning
    # =====================================================

    df = df.fillna(0)
    
    # Convert TotalCharges to double

    df = df.withColumn(
         "TotalCharges",
         col("TotalCharges").cast("double")
    )

    # Fill nulls after casting

    df = df.fillna({
         "TotalCharges": 0
    })

    # =====================================================
    # Create Simple Numeric Features
    # =====================================================

    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    assembler = VectorAssembler(
        inputCols=numeric_columns,
        outputCol="features"
    )

    final_df = assembler.transform(df)

    # =====================================================
    # Simulated Predictions
    # =====================================================

    print("\nGenerating Churn Predictions...\n")

    predictions_df = final_df.select(
        "customerID",
        "features"
    )

    # =====================================================
    # Simulate Churn Probability
    # =====================================================

    predictions_pd = predictions_df.toPandas()

    predictions_pd["churn_probability"] = (
        predictions_pd.index % 100
    ) / 100

    predictions_pd["prediction"] = predictions_pd[
        "churn_probability"
    ].apply(
        lambda x: 1 if x > 0.5 else 0
    )

    # =====================================================
    # Save Predictions
    # =====================================================

    output_path = (
        "data/predictions/"
        "daily_churn_predictions.csv"
    )

    predictions_pd.to_csv(
        output_path,
        index=False
    )

    print("\nPredictions Saved Successfully.")

    print(f"\nSaved To:\n{output_path}")

    # =====================================================
    # Show Sample Predictions
    # =====================================================

    print("\nSample Predictions:\n")

    print(
        predictions_pd.head()
    )

    # =====================================================
    # Save Batch Metadata
    # =====================================================

    batch_metadata = {

        "batch_job": "Daily Telecom Churn Prediction",

        "records_processed":
            int(len(predictions_pd)),

        "prediction_file":
            output_path
    }

    with open(
        "models/batch_job_metadata.json",
        "w"
    ) as f:

        json.dump(
            batch_metadata,
            f,
            indent=4
        )

    print("\nBatch Metadata Saved.")

    # =====================================================
    # Stop Spark
    # =====================================================

    spark.stop()

    print("\nWeek 4 Complete.\n")


if __name__ == "__main__":

    daily_batch_prediction()