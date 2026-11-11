# python # app.py - Streamlit UI: streamlit run app.py
import os
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

st.set_page_config(page_title="Techno Products Bot", page_icon="⚙️")
st.title("⚙️ Techno Products - Sales & Service Bot")
st.caption("Tamil + English | Ask like: 'M30 bolt price?' / 'மோட்டார் சத்தம் வருது'")

# Setup
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    db = QdrantVectorStore(
        client=client,
        collection_name="techno_products",
        embedding=embeddings
    )

    return db


db = load_db()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)





# Function
def get_answer(query):
    # 1. Retrieval
    docs = db.similarity_search(query, k=5)  # Top 5
    context = "\n\n".join([d.page_content for d in docs])

    # 2. Strict Prompt - No Hallucination
    prompt = f"""
    You are Techno Products Sales/Service Assistant.
    CONTEXT from catalog:
    {context}

    USER QUESTION: {query}

    Rules:
    1. Answer ONLY from context. If price not in context, say "Price list la illa, sales team kitta kekuren"
    2. If Tamil question, answer in Tamil. If English, English.
    3. Always give source page number if available.
    4. Never invent price or stock.
    """

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    return chat_completion.choices[0].message.content, docs


# UI
query = st.text_input("Ungal Kelvi:")
if query:
    with st.spinner("Catalog theduren..."):
        answer, sources = get_answer(query)
        st.success(answer)
        with st.expander("Source Documents"):
            for s in sources:
                st.write(s.page_content[:300])