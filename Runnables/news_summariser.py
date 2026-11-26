from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGroq(model= "openai/gpt-oss-20b")

search_tool = TavilySearch(max_results = 5)

prompt = ChatPromptTemplate.from_template(
     """
     You are a helpful assisstant. Summarise the folowing news into clear bullet points {news}
     """
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.invoke("Latest AI news of 2026")

result = chain.invoke({"news":news_result}) 

print(result)
