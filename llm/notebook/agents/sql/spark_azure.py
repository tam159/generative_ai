import os

from dotenv import load_dotenv
from pyspark import SparkConf
from pyspark.sql import SparkSession

load_dotenv()

storage_account_name = "datalakemwdvy04"
container_name = "files"
application_id = os.getenv("APPLICATION_ID", "")
tenant_id = os.getenv("TENANT_ID", "")
secret_value = os.getenv("SECRET_VALUE", "")

conf = SparkConf()
conf.set("spark.sql.warehouse.dir", "agent_warehouse")
conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
conf.set(
    "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
)

conf.set(
    "spark.jars.packages",
    "io.delta:delta-spark_2.13:3.2.0,org.apache.hadoop:hadoop-azure:3.4.0,org.apache.hadoop:hadoop-common:3.4.0",
)

conf.set(
    f"fs.azure.account.auth.type.{storage_account_name}.dfs.core.windows.net", "OAuth"
)
conf.set(
    f"fs.azure.account.oauth.provider.type.{storage_account_name}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
)
conf.set(
    f"fs.azure.account.oauth2.client.id.{storage_account_name}.dfs.core.windows.net",
    application_id,
)
conf.set(
    f"fs.azure.account.oauth2.client.secret.{storage_account_name}.dfs.core.windows.net",
    secret_value,
)
conf.set(
    f"fs.azure.account.oauth2.client.endpoint.{storage_account_name}.dfs.core.windows.net",
    f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
)

spark = (
    SparkSession.builder.config(conf=conf)
    .appName("spark-agent")
    .enableHiveSupport()
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# print(f"hadoop_version: {spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}")
# spark.sql("SHOW DATABASES").show()

df = spark.read.format("delta").load(
    f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/delta/products-delta"
)

df.show()
