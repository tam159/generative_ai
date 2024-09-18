"""Toolkit for interacting with Spark SQL."""


from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from spark_agent.spark_sql import SparkSQL
from spark_agent.spark_tool import (
    InfoSparkSQLTool,
    ListSparkSQLTool,
    QueryCheckerTool,
    QuerySparkSQLTool,
)


class SparkSQLToolkit(BaseToolkit):
    """
    Toolkit for interacting with Spark SQL.

    Parameters
    ----------
        db: SparkSQL. The Spark SQL database.
        llm: BaseLanguageModel. The language model.

    """

    db: SparkSQL = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_safe_tools(self) -> list[BaseTool]:
        """Get safe tools in the toolkit."""
        return [
            InfoSparkSQLTool(db=self.db),
            ListSparkSQLTool(db=self.db),
            QueryCheckerTool(db=self.db, llm=self.llm),
        ]

    def get_sensitive_tools(self) -> list[BaseTool]:
        """Get sensitive tools in the toolkit."""
        return [
            QuerySparkSQLTool(db=self.db),
        ]

    def get_tools(self) -> list[BaseTool]:
        """Get all tools in the toolkit."""
        return self.get_safe_tools() + self.get_sensitive_tools()
