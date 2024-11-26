import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool

TrinetraQ_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a TrinetraQ-Fraud detection bot. You detect fraud and risk transections in a bank "
)

# This is the message with which the system opens the conversation.
WELCOME_MSG = "Welcome to TrinetraQ Fraud detection bot. Type `q` to quit"
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


def chatbot(state: OrderState) -> OrderState:
    """The chatbot itself. A simple wrapper around the model's own chat interface."""
    message_history = [TrinetraQ_SYSINT] + state["messages"]
    return {"messages": [llm.invoke(message_history)]}


# Set up the initial graph based on our state definition.
graph_builder = StateGraph(OrderState)

# Add the chatbot function to the app graph as a node called "chatbot".
graph_builder.add_node("chatbot", chatbot)

# Define the chatbot node as the app entrypoint.
graph_builder.add_edge(START, "chatbot")

chat_graph = graph_builder.compile()
from pprint import pprint

user_msg = "Hello, what can you do?"
state = chat_graph.invoke({"messages": [user_msg]})
for msg in state["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")