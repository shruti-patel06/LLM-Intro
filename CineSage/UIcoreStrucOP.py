
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional

# Load env file
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

# Initialize Model
@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0.9
    )

model = get_model()

# Schema of the Output
class Movie(BaseModel):
    title: str
    director: str
    release_yr: Optional[int]
    genre: List[str]
    rating: Optional[float]
    cast: List[str]
    summary: str

# Initialize Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Create Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract movie information from the paragraph
        {format_instructions}"""
    ),
    ("human", "{paragraph}")
])

# Streamlit UI
st.title("🎬 Movie Information Extractor")

st.write(
    "Extract structured movie information from a paragraph "
    "using LangChain and Pydantic."
)

st.divider()

# Input from User
movie = st.text_area(
    "Enter Movie Description:",
    placeholder="Enter or paste the movie description here...",
    height=200
)

# Submit Button
if st.button("Extract Movie Details", use_container_width=True):

    if movie.strip():

        with st.spinner("Extracting movie details..."):

            # Format Prompt
            movie_prompt = prompt_template.invoke({
                "format_instructions": parser.get_format_instructions(),
                "paragraph": movie
            })

            # Get Response
            response = model.invoke(movie_prompt)

        st.subheader("🎥 Movie Details")

        st.write(response.content)

    else:
        st.warning("Please enter a movie description.")
