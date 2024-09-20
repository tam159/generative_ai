"""Lakehouse agent."""

from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from spark_agent.custom_exceptions import UnsupportedModelTypeError
from spark_agent.lakehouse_tool import (
    get_spark_system_message,
    spark_safe_tools,
    spark_sensitive_tools,
)

GPT_4O = "gpt-4o-2024-08-06"
GPT_4O_MINI = "gpt-4o-mini"


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> Runnable[LanguageModelInput, BaseMessage]:
    if model_name == "gpt-4o":
        model = ChatOpenAI(model=GPT_4O, temperature=0)
    elif model_name == "gpt-4o-mini":
        model = ChatOpenAI(model=GPT_4O_MINI, temperature=0)
    else:
        raise UnsupportedModelTypeError(model_name)

    return model.bind_tools(spark_safe_tools + spark_sensitive_tools)


class AgentState(MessagesState):
    """Agent state."""


class GraphConfig(TypedDict):
    """Graph configuration."""

    model_name: Literal["gpt-4o", "gpt-4o-mini"]


def should_continue(state: AgentState):
    """Determine whether to continue or not."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "end"

    return "continue"


def call_model(state: AgentState, config: RunnableConfig):
    """Call the model."""
    messages = state["messages"]

    last_human_message = None
    for message in messages[::-1]:
        if isinstance(message, HumanMessage):
            last_human_message = message
            break

    system_message = get_spark_system_message(last_human_message=last_human_message)
    messages = [system_message, *messages]

    model_name = config.get("configurable", {}).get("model_name", "gpt-4o")
    model = _get_model(model_name=model_name)
    response = model.invoke(messages)

    return {"messages": [response]}


def route_tools(
    state: AgentState,
) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
    """Route the tools."""
    next_node = tools_condition(state["messages"])

    if next_node == END:
        return "__end__"

    ai_message = state["messages"][-1]
    tool_calls = ai_message.tool_calls  # type: ignore # noqa: PGH003

    for tool_call in tool_calls:
        if tool_call["name"] in {t.name for t in spark_sensitive_tools}:
            return "sensitive_tools"

    return "safe_tools"


def get_compiled_graph() -> CompiledStateGraph:
    """Get the compiled graph."""
    workflow = StateGraph(state_schema=AgentState, config_schema=GraphConfig)

    workflow.add_node("agent", call_model)  # type: ignore # noqa: PGH003
    workflow.add_node("safe_tools", ToolNode(spark_safe_tools))
    workflow.add_node("sensitive_tools", ToolNode(spark_sensitive_tools))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        source="agent",
        path=route_tools,
    )

    workflow.add_edge("safe_tools", "agent")
    workflow.add_edge("sensitive_tools", "agent")

    return workflow.compile()
    # return workflow.compile(interrupt_before=["sensitive_tools"])


graph = get_compiled_graph()
