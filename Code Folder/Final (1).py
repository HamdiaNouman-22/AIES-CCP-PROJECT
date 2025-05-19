# Full Urdu RAG Chatbot with Conversation Memory, Caching, and Control Buttons
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from audio_recorder_streamlit import audio_recorder
import tempfile
import time
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize models and components
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    GROQ_API_KEY = "dummy_key"
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
    groq_model = ChatGroq(model_name="llama3-70b-8192", temperature=0.3, max_tokens=512)
    urdu_splitter = RecursiveCharacterTextSplitter(
        separators=["۔", "\n\n", "\n", "،", " "],
        chunk_size=400,
        chunk_overlap=100,
        length_function=len
    )
    return EMBED_MODEL, groq_model, urdu_splitter

def create_faiss_vectorstore(text, EMBED_MODEL):
    chunks = urdu_splitter.split_text(text[:100000])
    embeddings = EMBED_MODEL.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return {"chunks": chunks, "index": index}

# Initialize and cache memory and chain

def init_memory():
    return ConversationBufferMemory(
        memory_key="history",           # this must match the prompt
        input_key="input",              # tells LangChain which one is the main user input
        return_messages=True
    )


# Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "audio_cooldown" not in st.session_state:
    st.session_state.audio_cooldown = 0
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = init_memory()
if "show_memory" not in st.session_state:
    st.session_state.show_memory = False

# Load models
EMBED_MODEL, groq_model, urdu_splitter = load_models()

# Sidebar: PDF upload, Clear/Show Memory
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload Urdu PDF", type=["pdf"])
    if uploaded_file is not None and st.session_state.vectorstore is None:
        with st.spinner("Processing PDF..."):
            with BytesIO(uploaded_file.getvalue()) as pdf_data:
                doc = fitz.open(stream=pdf_data, filetype="pdf")
                raw_text = "".join([page.get_text() for page in doc])
                doc.close()
                st.session_state.vectorstore = create_faiss_vectorstore(raw_text, EMBED_MODEL)
                st.success("PDF loaded successfully!")

    if st.button("Clear Conversation Memory"):
        st.session_state.conversation_memory.clear()
        st.success("Conversation memory cleared!")
    st.sidebar.checkbox("Show Memory", key="show_memory")

# Core RAG + Memory + Cache + Chain
# Simplified signature: only question needed

def urdu_rag_query(question: str) -> str:
    # Return cached answer if available
    if question in st.session_state.qa_cache:
        return st.session_state.qa_cache[question]

    # Ensure vectorstore is loaded
    vectorstore = st.session_state.vectorstore
    # Retrieve context
    q_emb = EMBED_MODEL.encode([question])
    q_emb = np.array(q_emb).astype('float32')
    distances, indices = vectorstore["index"].search(q_emb, 15)
    context = "\n\n".join([vectorstore["chunks"][i] for i in indices[0]])

    # Initialize ConversationChain if needed
    if "conversation_chain" not in st.session_state:
        prompt = PromptTemplate(
            input_variables=["history", "input", "context"],
            template="""
            آپ کا کردار: ایک ماہر اردو استاد

            ہدایات:
            1. پچھلی گفتگو اور دیا گیا سیاق و سباق استعمال کریں۔
            2. اگر سوال مختصر اور معلوماتی ہو، تو مختصر جواب دیں۔
            3. اگر سوال میں وضاحت یا خلاصہ مانگا گیا ہو، تو تفصیلی جواب دیں۔
            4. جہاں ضروری ہو وہاں مشکل اردو الفاظ کی وضاحت کریں۔
            5. جواب کے شروع میں 'جواب:' نہ لکھیں۔

            پچھلی گفتگو:
            {history}

            سیاق:
            {context}

            سوال:
            {input}

            جواب:
            """
        )
        st.session_state.conversation_chain = LLMChain(
            llm=groq_model,
            prompt=prompt,
            memory=st.session_state.conversation_memory,
            verbose=False
        )

    # Build history string from memory
    history = "\n".join([
        f"انسان: {m.content}" if m.type == "human" else f"AI: {m.content}"
        for m in st.session_state.conversation_memory.chat_memory.messages
    ])

    # Predict with chain, passing context and history
    resp = st.session_state.conversation_chain.predict(
        input=question,
        context=context
    )

    # Cache and return
    st.session_state.qa_cache[question] = resp
    return resp

# Text-to-speech helper
def text_to_speech(text: str, lang: str = "ur") -> BytesIO:
    tts = gTTS(text=text, lang=lang, slow=False)
    b = BytesIO(); tts.write_to_fp(b); b.seek(0)
    return b

# Main UI
st.title("📚 Urdu Tutor")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(
            f'<div style="text-align: right; direction: rtl;">{msg["content"]}</div>',
            unsafe_allow_html=True
        )
        if msg.get("audio"): st.audio(msg["audio"], format="audio/mp3")

# Show memory buffer if toggled
if st.session_state.show_memory:
    st.subheader("Conversation Memory")
    st.code(st.session_state.conversation_memory.buffer_as_str, language="text")

# Input controls
t1, t2 = st.columns([5,1])
with t1:
    user_input = st.chat_input("Type your question...", key="input")
with t2:
    audio_bytes = None
    if time.time() - st.session_state.audio_cooldown > 3:
        audio_bytes = audio_recorder(text="", pause_threshold=2.5, key="recorder")
    else:
        st.warning("Please wait...")

# Process voice input
current = time.time()
if audio_bytes and len(audio_bytes)>0 and not st.session_state.processing and current - st.session_state.audio_cooldown>3:
    st.session_state.processing=True; st.session_state.audio_cooldown=current
    with st.spinner("Transcribing..."):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes); path=tmp.name
            r=sr.Recognizer();
            with sr.AudioFile(path) as source:
                audio=r.record(source); text=r.recognize_google(audio, language="ur-PK")
            os.remove(path)
            if text and text!=st.session_state.last_processed:
                st.session_state.messages.append({"role":"user","content":text})
                st.session_state.last_processed=text
                if st.session_state.vectorstore:
                    ans=urdu_rag_query(text)
                    audio=text_to_speech(ans)
                    st.session_state.messages.append({"role":"assistant","content":ans,"audio":audio})
                    st.rerun()
                else: st.warning("Upload a PDF first.")
        except Exception as e:
            st.error(f"Error: {e}")
        finally: st.session_state.processing=False

# Process text input
if user_input and not st.session_state.processing and user_input!=st.session_state.last_processed:
    st.session_state.processing=True
    st.session_state.messages.append({"role":"user","content":user_input})
    st.session_state.last_processed=user_input
    if st.session_state.vectorstore:
        with st.spinner("Generating answer..."):
            ans=urdu_rag_query(user_input)
            audio=text_to_speech(ans)
            st.session_state.messages.append({"role":"assistant","content":ans,"audio":audio})
    else:
        st.warning("Upload a PDF first.")
    st.session_state.processing=False
    st.rerun()
