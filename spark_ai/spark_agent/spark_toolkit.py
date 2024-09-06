"""Toolkit for interacting with Spark SQL."""

from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from spark_agent.spark_tool import (
    InfoSparkSQLTool,
    ListSparkSQLTool,
    QueryCheckerTool,
    QuerySparkSQLTool,
)
from spark_agent.spark_sql import SparkSQL


class SparkSQLToolkit(BaseToolkit):
    """Toolkit for interacting with Spark SQL.

    Parameters:
        db: SparkSQL. The Spark SQL database.
        llm: BaseLanguageModel. The language model.
    """

    db: SparkSQL = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_safe_tools(self) -> List[BaseTool]:
        """Get safe tools in the toolkit."""
        return [
            InfoSparkSQLTool(db=self.db),
            ListSparkSQLTool(db=self.db),
            QueryCheckerTool(db=self.db, llm=self.llm),
        ]

    def get_sensitive_tools(self) -> List[BaseTool]:
        """Get sensitive tools in the toolkit."""
        return [
            QuerySparkSQLTool(db=self.db),
        ]

    def get_tools(self) -> List[BaseTool]:
        """Get all tools in the toolkit."""
        return self.get_safe_tools() + self.get_sensitive_tools()
