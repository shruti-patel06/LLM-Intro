from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

model = init_chat_model("openai/gpt-oss-120b", model_provider="groq",temperature=0.9)

parser = StrOutputParser()

# Step-by Step manual flow

# formatted_prompt = prompt.format_messages(topic="Machine Learning")

# # Call the model Manually
# response=model.invoke(formatted_prompt)

# print(response.content)


# Chain Concept using Runnables

chain = prompt | model | parser

response = chain.invoke("Deep Learning")

print(response)