"""streamlit providing the UI functionality."""
import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate


# load llama2 model
@st.cache_resource
def load_model():
    """load the llama2 model"""

    llama2_model = "TheBloke/Llama-2-7B-Chat-GGML"
    model_variant = "llama-2-7b-chat.ggmlv3.q8_0.bin"
    model = CTransformers(
        model=llama2_model,
        model_file=model_variant,
        config={"max_new_tokens": 256, "temperature": 0},
    )
    return model


# request for blog content
@st.cache_data
def generate_blog(query, word_count, genre):
    """generate blog"""

    template = """
    As a you a blogger, please write a blog for the topic {query} and under gener {genre}.
    Please write the blog in {word_count} words.
    Your blog strictly should be in markdown language.
    Also, come with an appropriate title for the blog and set it at the top in bold letters.
    
    Example: 
    ## **YOUR TITLE**    
    YOUR BLOG COME HERE     
    """

    prompt = PromptTemplate(
        input_variables=["query", "word_count", "genre"], template=template
    )

    llama2_model = load_model()
    response = llama2_model(
        prompt.format(query=query, word_count=word_count, genre=genre)
    )
    return response


st.set_page_config(
    page_title="The Blogger",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("The Blogger 📝")

user_input = st.text_input("Enter the topic you wanted the blog")

col_1, col_2 = st.columns([5, 5])
with col_1:
    no_of_words = st.selectbox(
        "Number of words", ("200", "250", "300", "350", "400", "500")
    )
with col_2:
    blog_genres = st.selectbox(
        "Genre of the blog",
        (
            "Business",
            "Food",
            "Music",
            "Fitness",
            "Travel",
            "Gaming",
            "Finance",
            "Sports",
            "Movie",
            "Science & Technology",
            "Lifestyle",
        ),
        index=0,
    )

submit = st.button("Blog")
if submit:
    blog = generate_blog(user_input, no_of_words, blog_genres)
    st.write(blog)
