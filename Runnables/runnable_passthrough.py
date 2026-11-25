from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

load_dotenv()

code_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a Code Generator."),
    ("human","{topic}")
])
explain_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a Code Explainer who explains the code in simple language."),
    ("human","Explain the following code in simple terms {code}")
])

model = init_chat_model("openai/gpt-oss-120b", model_provider="groq",temperature=0.4)

parser = StrOutputParser()

# seq = code_prompt | model | parser | explain_prompt | model | parser --> this doesnot give the code, it starts explaining directly.
# seq = code_prompt | model | parser | RunnableLambda(lambda code: {"code": code}) | explain_prompt | model | parser --> StrOutputParser() converts the LLM response into a plain Python string.Your explain_prompt expects a dictionary with a key named code.Passing a string directly doesn't satisfy that input requirement.

seq1 = code_prompt | model | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(), #This returns the result same as the input here it is code from seq1
    "explaination" : explain_prompt | model | parser # Gives explaination of the code from seq1
})

chain = seq1 | seq2

response = chain.invoke({"topic":"write a simple Palindrome code in simple python."})

print(response['code'])
print(response['explaination'])