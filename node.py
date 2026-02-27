from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from react import tools, llm

load_dotenv()


SYSTEM_MESSAGE = "You are a helpful assistant that can use tools to answer questions."

def reasoning_node(state: MessagesState) -> MessagesState:
    """
    This node takes in the conversation history and generates a response using the LLM.
    It can also decide to call tools if needed.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)