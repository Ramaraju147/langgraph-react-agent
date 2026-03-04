from dotenv import load_dotenv;
import os
from langchain.messages import HumanMessage
from langgraph.graph import MessagesState,StateGraph, END
from node import reasoning_node, tool_node

load_dotenv()

flow = StateGraph(MessagesState)

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> bool:
    """
    This function checks if the agent has made any tool calls in the last message.
    If it has, we should continue to the ACT node to execute those tool calls.
    If not, we can end the flow.
    """
    if state["messages"][LAST].tool_calls:
        return ACT
    return END
flow.add_node(AGENT_REASON,reasoning_node)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT,tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    ACT: ACT,
    END: END
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in France? and triple temperature for me!")]})
    print(res["messages"][LAST].content) 
