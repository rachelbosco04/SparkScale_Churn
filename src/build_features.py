import sys
import os

sys.path.append(os.path.dirname(__file__))

from spark_config import create_spark_session

from pyspark.sql.functions import avg, count

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml import Pipeline


def build_features():

    # =====================================================
    # Create Spark Session
    # =====================================================

    spark = create_spark_session()

    print("\nLoading cleaned dataset...\n")

    # =====================================================
    # Load cleaned parquet
    # =====================================================

    df = spark.read.parquet(
        "data/processed/cleaned_telco.parquet"
    )

    print(f"Rows Loaded: {df.count()}")

    # =====================================================
    # Register temp SQL table
    # =====================================================

    df.createOrReplaceTempView(
        "telco_customers"
    )

    # =====================================================
    # Spark SQL Aggregations
    # =====================================================

    print("\nRunning Spark SQL Aggregations...\n")

    agg_df = spark.sql("""

        SELECT

            Contract,

            AVG(MonthlyCharges) AS avg_monthly_charge,

            AVG(TotalCharges) AS avg_total_charge,

            COUNT(*) AS customer_count,

            AVG(tenure) AS avg_tenure

        FROM telco_customers

        GROUP BY Contract

    """)

    agg_df.show()

    # =====================================================
    # Categorical Columns
    # =====================================================

    categorical_columns = [

        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"

    ]

    # =====================================================
    # Numeric Columns
    # =====================================================

    numeric_columns = [

        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"

    ]

    # =====================================================
    # String Indexers
    # =====================================================

    indexers = [

        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_index",
            handleInvalid="keep"
        )

        for column in categorical_columns

    ]

    # =====================================================
    # One Hot Encoders
    # =====================================================

    encoders = [

        OneHotEncoder(
            inputCol=f"{column}_index",
            outputCol=f"{column}_encoded"
        )

        for column in categorical_columns

    ]

    # =====================================================
    # Feature Columns
    # =====================================================

    feature_columns = [

        f"{column}_encoded"

        for column in categorical_columns

    ] + numeric_columns

    # =====================================================
    # Vector Assembler
    # =====================================================

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )

    # =====================================================
    # ML Pipeline
    # =====================================================

    pipeline = Pipeline(
        stages=indexers + encoders + [assembler]
    )

    print("\nFitting Feature Pipeline...\n")

    pipeline_model = pipeline.fit(df)

    transformed_df = pipeline_model.transform(df)

    # =====================================================
    # Final Feature Dataset
    # =====================================================

    final_df = transformed_df.select(
        "features",
        "label"
    )

    print("\nFinal Feature Dataset:\n")

    final_df.show(5, truncate=False)

    # =====================================================
    # Train/Test Split
    # =====================================================

    train_df, test_df = final_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print(f"\nTrain Rows: {train_df.count()}")
    print(f"Test Rows: {test_df.count()}")


    # =====================================================
    # Save Feature Datasets
    # =====================================================

    train_output = "data/processed/train_data.pkl"

    test_output = "data/processed/test_data.pkl"

    # Spark -> Pandas
    train_pd = train_df.toPandas()

    test_pd = test_df.toPandas()

    # Save as pickle
    train_pd.to_pickle(train_output)

    test_pd.to_pickle(test_output)

    print("\nFeature Engineering Complete.")

    print(f"\nTrain dataset saved to:\n{train_output}")

    print(f"\nTest dataset saved to:\n{test_output}")
    
    
    # =====================================================
    # DAG Visualization
    # =====================================================

    print("\nSpark Execution Plan:\n")

    final_df.explain(True)

    return train_df, test_df


if __name__ == "__main__":

    build_features()