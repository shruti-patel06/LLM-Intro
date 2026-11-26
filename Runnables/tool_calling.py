from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from rich import print

load_dotenv()

# 1.Creating a tool
@tool
def get_text_length(text:str)->int:
    """Returns the number of character in a given text"""
    return len(text)

llm = ChatGroq(model = "openai/gpt-oss-20b")

# 2.Tool Binding

llm_with_tool = llm.bind_tools([get_text_length])

# result1 = llm.invoke("hello")

# 3.Tool Calling
result = llm_with_tool.invoke("Calculate the length of the word hello")

print(result)

# 4.Tool Execution
if result.tool_calls:
    for tool_call in result.tool_calls:
        output = get_text_length.invoke(tool_call["args"])
        print("Tool output:",output)