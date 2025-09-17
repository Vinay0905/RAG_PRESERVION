import os
import platform
import logging
import tempfile
import re
import contextlib
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
import edge_tts

# CONFIG & Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCUMENT_URLS = os.getenv("DOCUMENT_URLS", "W:/anaconda/18-5-25/DOCUMENTATION.pdf")

def initialize_llm():
    try:
        logging.info("Initializing primary LLM model...")
        return ChatOpenAI(model="gpt-3.5-turbo")
    except Exception as e:
        logging.error(f"Failed to initialize primary model: {e}")
        raise

llm = initialize_llm()

template = """
You are a helpful *website navigation assistant*.
You know the structure of the website based on the documentation provided.
Your job is to guide the user step by step, as if you are showing them the way through the website.
User Question:
{question}
Relevant Information from Documentation:
{context}
Answer clearly, in a guiding tone (e.g., "First, go to the homepage... then click on Farmer Dashboard...").
⚠ IMPORTANT: The user has selected a preferred language.
Always provide your *final answer fully in {language}*.
The supported languages are Hindi, Tamil, and English.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question", "language"]
)

class RAG:
    def __init__(self, urls):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstore = InMemoryVectorStore(self.embeddings)
        self.prompt = prompt
        try:
            loader = PyPDFLoader(urls)
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(docs)
            self.vectorstore.add_documents(documents=docs)
        except Exception as e:
            logging.error(f"Error loading or processing documents: {e}")
            raise

    def retrieve(self, question):
        return self.vectorstore.similarity_search(question)

    def generate(self, question, context, language):
        docs_content = "\n\n".join(doc.page_content for doc in context)
        messages = self.prompt.invoke({
            "question": question,
            "context": docs_content,
            "language": language
        })
        try:
            response = llm.invoke(messages, max_tokens=1024, temperature=0.7)
            return response.content
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}")
            return "Sorry, I encountered an error while generating the answer."

class Audio_Generation:
    file_index = 0
    _audio_files = set()
    @staticmethod
    @contextlib.contextmanager
    def temp_audio_file(suffix=".mp3"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            yield tf.name
    @classmethod
    def play_audio_blob(cls, text: str, language="english"):
        if not text or not text.strip():
            return None
        voice_map = {
            "english": "en-US-AriaNeural",
            "hindi": "hi-IN-SwaraNeural",
            "tamil": "ta-IN-PallaviNeural",
        }
        voice = voice_map.get(language.lower(), "en-US-AriaNeural")
        try:
            cls.file_index += 1
            with cls.temp_audio_file() as fname:
                communicate = edge_tts.Communicate(text, voice)
                asyncio.run(communicate.save(fname))
                cls._audio_files.add(fname)
                return fname
        except Exception as e:
            logging.error(f"Error generating Edge TTS audio: {e}")
            return None
    @classmethod
    def cleanup(cls):
        for fname in cls._audio_files:
            try:
                if os.path.exists(fname):
                    os.remove(fname)
                    logging.info(f"Cleaned up: {fname}")
            except Exception as e:
                logging.error(f"Error cleaning up {fname}: {e}")
        cls._audio_files.clear()

# --- FASTAPI APP STARTS HERE ---
app = FastAPI(title="Preservion Voice RAG API")

# Global instance (for demo: rebuild on startup)
rag = RAG(urls=DOCUMENT_URLS)

class QueryRequest(BaseModel):
    question: str
    language: str = "english"

@app.post("/ask")
async def ask_question(body: QueryRequest):
    """API endpoint for answering questions using RAG and LLM."""
    context = rag.retrieve(body.question)
    answer = rag.generate(body.question, context, body.language)
    return {"answer": answer}

@app.post("/tts/")
async def tts(text: str = Form(...), language: str = Form("english")):
    """Text-to-Speech conversion endpoint."""
    audio_file = Audio_Generation.play_audio_blob(text, language)
    if audio_file and os.path.exists(audio_file):
        # Serve the audio file as bytes
        with open(audio_file, "rb") as file_data:
            contents = file_data.read()
        return JSONResponse(content={"audio_base64": base64.b64encode(contents).decode()})
    return JSONResponse(content={"error": "Failed to generate audio"}, status_code=500)

@app.get("/")
def root():
    return {"message": "Preservion Voice RAG API is running!"}

# Optional: Add endpoints for file upload, conversation, etc. as needed
