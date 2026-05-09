import os
import tempfile

from pyspark.sql import SparkSession


# =====================================================
# Environment Variables
# =====================================================

os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot"

os.environ["PYSPARK_PYTHON"] = r"C:\Users\Lenovo\Desktop\SparkScale_Churn\venv\Scripts\python.exe"

os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\Lenovo\Desktop\SparkScale_Churn\venv\Scripts\python.exe"

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"

os.environ["PATH"] += r";C:\hadoop\bin"

# =====================================================
# Temp Directory
# =====================================================

temp_dir = tempfile.mkdtemp()

os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir


def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("SparkScaleChurn")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.warehouse.dir", temp_dir)
        .config("spark.local.dir", temp_dir)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark