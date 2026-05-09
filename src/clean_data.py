from pyspark.sql.functions import col, when, trim
from spark_config import create_spark_session
from ingest_data import load_raw_data


def clean_data():

    # Load Spark Session
    spark = create_spark_session()

    # Load Raw Dataset
    df = load_raw_data()

    print("\nStarting Data Cleaning Pipeline...\n")

    # =====================================================
    # Remove Duplicate Rows
    # =====================================================

    before = df.count()

    df = df.dropDuplicates()

    after = df.count()

    print(f"Duplicates Removed: {before - after}")

    # =====================================================
    # Trim Spaces from String Columns
    # =====================================================

    string_columns = [
        field.name
        for field in df.schema.fields
        if field.dataType.simpleString() == "string"
    ]

    for column_name in string_columns:
        df = df.withColumn(
            column_name,
            trim(col(column_name))
        )

    # =====================================================
    # Clean TotalCharges Column
    # =====================================================

    df = df.withColumn(
        "TotalCharges",
        when(
            col("TotalCharges") == "",
            None
        ).otherwise(col("TotalCharges"))
    )

    # Convert to Double
    df = df.withColumn(
        "TotalCharges",
        col("TotalCharges").cast("double")
    )

    # =====================================================
    # Missing Value Analysis
    # =====================================================

    print("\nMissing Values:\n")

    for column_name in df.columns:

        missing_count = df.filter(
            col(column_name).isNull()
        ).count()

        print(f"{column_name}: {missing_count}")

    # =====================================================
    # Fill Missing Values
    # =====================================================

    avg_total = df.selectExpr(
        "avg(TotalCharges)"
    ).collect()[0][0]

    df = df.fillna({
        "TotalCharges": avg_total
    })

    # =====================================================
    # Create Label Column
    # =====================================================

    df = df.withColumn(
        "label",
        when(col("Churn") == "Yes", 1).otherwise(0)
    )

    # =====================================================
    # Drop Unnecessary Columns
    # =====================================================

    df = df.drop("customerID")

    # =====================================================
    # Final Dataset Info
    # =====================================================

    print("\nCleaned Dataset Schema:\n")

    df.printSchema()

    print("\nSample Cleaned Data:\n")

    df.show(5, truncate=False)

    print(f"\nFinal Row Count: {df.count()}")

    # =====================================================
    # Save Cleaned Dataset
    # =====================================================

    output_path = "data/processed/cleaned_telco.parquet"

    df.toPandas().to_parquet(
    output_path,
    index=False
)

    print(f"\nCleaned parquet saved to:\n{output_path}")

    return df


if __name__ == "__main__":

    clean_data()