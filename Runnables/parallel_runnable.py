from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

load_dotenv()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines."
)

long_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail."
)

model = init_chat_model("openai/gpt-oss-120b", model_provider="groq",temperature=0.4)

parser = StrOutputParser()


# topic = "Machine Learning"

# formatted_short = short_prompt.format_messages(topic = topic)
# response_short = model.invoke("Generative AI")
# str_output_short = parser.parse(response_short.content)

# formatted_long = long_prompt.format_messages(topic = topic)
# response_long = model.invoke("Generative AI")
# str_output_long = parser.parse(response_long.content)

# print(str_output_short)
# print(str_output_long)



chain = RunnableParallel({
    "short" : RunnableLambda(lambda x : x['short']) | short_prompt | model | parser,
    "long" : RunnableLambda(lambda x : x['long']) | long_prompt |model | parser
})

# response = chain.invoke({"topic": "Reinforcement learning"})

# If LambdaRunnable is not used then,entire dictionary is passed at once to the chains which gives input error -->invoke({short...,long...})
response = chain.invoke({
    "short":{"topic":"Deep Learning"},
    "long" : {"topic": "Machine Learning"}
})

print(response['short'])
print(response['long'])
