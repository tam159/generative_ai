"""Lakehouse tools and utilities."""

import ast
import re

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools.simple import Tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, regexp_replace, to_date
from pyspark.sql.types import FloatType, IntegerType

from spark_agent.spark_prompt import (
    few_shot_examples,
    few_shot_header,
    search_proper_nouns_tool_description,
    search_proper_nouns_tool_name,
    system_prompt,
)
from spark_agent.spark_sql import SparkSQL
from spark_agent.spark_toolkit import SparkSQLToolkit

APP_NAME = "lakehouse_agent"
CATALOG = "spark_catalog"
SCHEMA = "salesforce"
DATA_PATH = "spark_agent/data"
GPT_4O = "gpt-4o-2024-08-06"
TEXT_EMBEDDING = "text-embedding-3-small"


llm_model = ChatOpenAI(model=GPT_4O, temperature=0)


def get_spark_conf():
    """Get the Spark configuration."""
    conf = SparkConf()
    conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", False)
    conf.set("spark.sql.broadcastTimeout", "2400")
    conf.set("spark.sql.warehouse.dir", "agent_warehouse")
    conf.set(
        "spark.jars.packages",
        "io.delta:delta-spark_2.13:3.2.0",
    )
    conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    conf.set(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    return conf


def get_spark_session(app_name: str, conf: SparkConf) -> SparkSession:
    """Get the Spark session."""
    spark_session = (
        SparkSession.builder.config(conf=conf)
        .appName(app_name)
        .enableHiveSupport()
        .getOrCreate()
    )

    spark_session.sparkContext.setLogLevel("ERROR")

    return spark_session


def get_spark_db(
    spark_session: SparkSession,
    catalog: str,
    schema: str,
    sample_rows_in_table_info: int = 10,
) -> SparkSQL:
    """Get the Spark database."""
    return SparkSQL(
        spark_session=spark_session,
        catalog=catalog,
        schema=schema,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )


def to_snake_case(col_name):
    """Convert a column name to snake case."""
    if col_name.lower() in ["CompanyEXTID".lower(), "CompanEXTID".lower()]:
        return "company_ext_id"

    return col_name.lower().replace(" ", "_")


class SalesforceLoad:
    """Load Salesforce data into the lakehouse."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        spark_session.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")

    def load_accounts(self):
        """Load the accounts data."""
        account_df = self.spark.read.load(
            f"{DATA_PATH}/Accounts.csv", format="csv", header=True, lineSep="\r"
        )
        account_df = account_df.toDF(
            *[to_snake_case(column) for column in account_df.columns]
        )
        account_df = account_df.withColumn(
            "company_ext_id", col("company_ext_id").cast(IntegerType())
        )
        account_df.write.format("delta").mode("overwrite").saveAsTable(
            f"{SCHEMA}.accounts"
        )

    def load_contacts(self):
        """Load the contacts data."""
        contact_df = self.spark.read.load(
            f"{DATA_PATH}/Contacts.csv", format="csv", header=True, lineSep="\r"
        )
        contact_df = contact_df.toDF(
            *[to_snake_case(column) for column in contact_df.columns]
        )
        contact_df = contact_df.withColumn(
            "company_ext_id", col("company_ext_id").cast(IntegerType())
        )
        contact_df.write.format("delta").mode("overwrite").saveAsTable(
            f"{SCHEMA}.contacts"
        )

    def load_leads(self):
        """Load the leads data."""
        lead_df = self.spark.read.load(
            f"{DATA_PATH}/Leads.csv", format="csv", header=True, lineSep="\r"
        )
        lead_df = lead_df.toDF(*[to_snake_case(column) for column in lead_df.columns])
        lead_df.write.format("delta").mode("overwrite").saveAsTable(f"{SCHEMA}.leads")

    def load_opportunities(self):
        """Load the opportunities data."""
        opportunity_df = self.spark.read.load(
            f"{DATA_PATH}/Opportunities.csv", format="csv", header=True, lineSep="\r"
        )
        opportunity_df = opportunity_df.toDF(
            *[to_snake_case(column) for column in opportunity_df.columns]
        )
        opportunity_df = (
            opportunity_df.withColumn(
                "company_ext_id", col("company_ext_id").cast(IntegerType())
            )
            .withColumn("close_date", to_date(col("close_date"), "M/d/yy"))
            .withColumn(
                "amount", regexp_replace(col("amount"), "[$,]", "").cast(FloatType())
            )
            .withColumn("currency", lit("USD"))
        )
        opportunity_df.write.format("delta").mode("overwrite").saveAsTable(
            f"{SCHEMA}.opportunities"
        )

    def load_data(self):
        """Load all data."""
        tables_loads = {
            "accounts": self.load_accounts,
            "contacts": self.load_contacts,
            "leads": self.load_leads,
            "opportunities": self.load_opportunities,
        }

        tables = self.spark.catalog.listTables(SCHEMA)
        table_names = [table.name for table in tables]

        for table_name, load_func in tables_loads.items():
            if table_name not in table_names:
                load_func()


def query_as_list(db: SparkSQL, query: str) -> list[str]:
    """Query as a list."""
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]

    return list(set(res))


def get_noun_retriever(
    nouns: list[str], embedding_model: str = TEXT_EMBEDDING, k: int = 3
) -> VectorStoreRetriever:
    """Get a retriever for the nouns."""
    vector_db = FAISS.from_texts(nouns, OpenAIEmbeddings(model=embedding_model))
    return vector_db.as_retriever(search_kwargs={"k": k})


def get_noun_retriever_tool(nouns: list[str]) -> Tool:
    """Get a tool for retrieving nouns."""
    noun_retriever = get_noun_retriever(nouns=nouns)
    return create_retriever_tool(
        noun_retriever,
        name=search_proper_nouns_tool_name,
        description=search_proper_nouns_tool_description,
    )


few_shot_example_selector = SemanticSimilarityExampleSelector.from_examples(
    few_shot_examples,
    OpenAIEmbeddings(model=TEXT_EMBEDDING),
    FAISS,
    k=3,
    input_keys=["input"],
)


def get_spark_system_message(last_human_message: HumanMessage | None) -> SystemMessage:
    """Get the system message for the Spark SQL agent."""
    spark_system_prompt = system_prompt.format(tables=spark_db.get_usable_table_names())

    if last_human_message:
        retrieved_examples: list[dict] = few_shot_example_selector.select_examples(
            input_variables={"input": last_human_message.content}  # type: ignore # noqa: PGH003
        )
        if retrieved_examples:
            examples = [
                f"User input: {retrieved_example.get('input')}\nSQL query: {retrieved_example.get('query')}"
                for retrieved_example in retrieved_examples
            ]
            spark_system_prompt += few_shot_header + "\n".join(examples)

    return SystemMessage(content=spark_system_prompt)


spark_conf = get_spark_conf()
spark_session = get_spark_session(app_name=APP_NAME, conf=spark_conf)

salesforce_loader = SalesforceLoad(spark=spark_session)
salesforce_loader.load_data()

spark_db = get_spark_db(spark_session=spark_session, catalog=CATALOG, schema=SCHEMA)
spark_toolkit = SparkSQLToolkit(db=spark_db, llm=llm_model)

spark_safe_tools = spark_toolkit.get_safe_tools()
spark_sensitive_tools = spark_toolkit.get_sensitive_tools()

company_names = query_as_list(
    spark_db, "SELECT company_name FROM accounts UNION SELECT company FROM leads"
)

noun_retriever_tool = get_noun_retriever_tool(nouns=company_names)
spark_safe_tools.append(noun_retriever_tool)
