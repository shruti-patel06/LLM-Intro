from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List,Optional

# Load env file
load_dotenv()

# Initialize Model
model = init_chat_model("openai/gpt-oss-120b", model_provider="groq",temperature=0.9)

# Schema of the Output
class Movie(BaseModel):
    title:str
    director:str
    release_yr:Optional[int]
    genre:List[str]
    rating:Optional[float]
    cast:List[str]
    summary:str

parser = PydanticOutputParser(pydantic_object = Movie)

# Create Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
('system',"""Extract movie information from the paragraph {format_instructions}"""),
('human',"{paragraph}")

])

# Input from User
movie = input("Enter movie Description:------>\n")

# Format Prompt
movie_prompt = prompt_template.invoke(
    {"format_instructions":parser.get_format_instructions(),
     "paragraph":movie
    })

# Get Response
response = model.invoke(movie_prompt)
movie_data = parser.parse(response.content)
print("\n Movie Details")
print(movie_data)