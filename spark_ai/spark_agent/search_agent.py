"""Search Agent. This agent is a simple agent that uses the TavilySearch tool to search for information."""

from functools import lru_cache
from typing import Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode


class UnsupportedModelTypeError(ValueError):
    """Unsupported model type error."""

    def __init__(self, model_name: str):
        super().__init__(f"Unsupported model type: {model_name}")


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> Runnable[LanguageModelInput, BaseMessage]:
    if model_name == "gpt-4o":
        model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
    elif model_name == "gpt-4o-mini":
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise UnsupportedModelTypeError(model_name)

    return model.bind_tools(tools)


class AgentState(MessagesState):
    """Agent state."""

    is_last_step: IsLastStep


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


system_prompt = """Be a helpful assistant"""
system_message = SystemMessage(content=system_prompt)


def call_model(state: AgentState, config: RunnableConfig):
    """Call the model."""
    messages = state["messages"]
    messages = [system_message, *messages]
    model_name = config.get("configurable", {}).get("model_name", "gpt-4o-mini")
    model = _get_model(model_name=model_name)
    response = model.invoke(messages)

    if (
        state["is_last_step"]
        and isinstance(response, AIMessage)
        and response.tool_calls
    ):
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }

    return {"messages": [response]}


tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools)


workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("agent", call_model)  # type: ignore # noqa: PGH003
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    source="agent",
    path=should_continue,
    path_map={
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()
