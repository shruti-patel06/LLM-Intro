from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model("openai/gpt-oss-120b", model_provider="groq",temperature=0.9,max_tokens=50)
response = model.invoke("Write a poem on AI.")
print(response.content)



# model = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0.3)

# response = model.invoke("Write a poem on AI.")

# if isinstance(response.content, str):
#     print(response.content)
# else:
#     for block in response.content:
#         if block.get("type") == "text":
#             print(block["text"])
