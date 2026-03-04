from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv


load_dotenv()

@tool
def triple(number: float) -> float:
    """
    Parameters: number (float): The number to be tripled.
    Returns the triple of a number.'
    """
    return number * 3

tools = [TavilySearch(max_results=2), triple]

llm = ChatOpenAI(model="gpt-4-0613", temperature=0).bind_tools(tools)