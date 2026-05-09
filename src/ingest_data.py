from spark_config import create_spark_session


def load_raw_data():

    spark = create_spark_session()

    df = spark.read.csv(
        "data/raw/telco_churn.csv",
        header=True,
        inferSchema=True
    )

    print("\n Dataset Loaded Successfully")
    print(f"Total Rows: {df.count()}")
    print(f"Total Columns: {len(df.columns)}\n")

    df.printSchema()

    return df


if __name__ == "__main__":

    df = load_raw_data()

    df.show(5)