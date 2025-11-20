import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import os
from llama_cpp import Llama
from spellchecker import SpellChecker
import re

# ------------------- CONFIG -------------------
MODEL_PATH = r"D:\gpt4all-models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MAX_TOKENS = 1200
N_CTX = 2048
TOP_K = 3

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}. Please check the path.")
    st.stop()

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("advanced-dataset.csv").fillna("")
    df['intent_text'] = (
        df['Problem Statement'] + ' ' +
        df['Advantages'] + ' ' +
        df['Target Audience'] + ' ' +
        df['Tech Stack Required'] + ' ' +
        df['Business Model Type'] + ' ' +
        df['Budget Range']
    )
    return df

df = load_data()

# ------------------- BUILD INDEX -------------------
@st.cache_resource
def build_index(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray().astype("float32")
    index = faiss.IndexFlatL2(X_dense.shape[1])
    index.add(X_dense)
    return vectorizer, index

vectorizer, index = build_index(df['intent_text'])

# ------------------- SEARCH -------------------
def get_similar_startups(query, top_k=TOP_K):
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    _, idx = index.search(query_vec, top_k)
    return df.iloc[idx[0]]

# ------------------- LOAD LLM -------------------
@st.cache_resource
def load_llm():
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=20,
            n_threads=6,
            n_batch=256,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ------------------- SPELL CHECK -------------------
@st.cache_resource
def load_spell_checker():
    return SpellChecker()

def correct_text(text):
    try:
        spell = load_spell_checker()
        words = text.split()
        corrected_words = [spell.correction(word) or word for word in words]
        return " ".join(corrected_words)
    except:
        return text

# ------------------- FORMAT AI OUTPUT -------------------
def format_ai_output(text):
    """
    Clean and format LLM output into readable startup ideas with proper sections.
    """
    # Map common variants / typos to standard sections with emojis
    sections = {
        r'project name[:\s]*': '💡  Project Name-: ',
        r'problem[:\s]*': '🛑  Problem-: ',
        r'solution[:\s]*': '✅  Solution-: ',
        r'target audience[:\s]*': '🎯  Target Audience-: ',
        r'tech stack[:\s]*': '💻  Tech Stack-: ',
        r'business model[:\s]*': '💰  Business Model-: ',
        r'budget[:\s]*': '💵  Budget-: '
    }

    formatted_text = text
    for pattern, replacement in sections.items():
        formatted_text = re.sub(pattern, f'\n\n{replacement}', formatted_text, flags=re.IGNORECASE)

    # Remove extra ** symbols
    formatted_text = re.sub(r'\*+', '', formatted_text)

    # Fix multiple newlines
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)

    return formatted_text.strip()

# ------------------- LOAD RESOURCES -------------------
llm = load_llm()
if llm is None:
    st.stop()

# ------------------- STREAMLIT UI -------------------
st.title("🚀 WEL-COME to Idea-bot")
st.info("💻 A New AI-based Startup Idea Generatoin model")

query = st.text_input("💡 Describe your startup goal, audience, or tech:")

if query:
    st.write("🔍 Finding inspiration from our knowledge base...")
    results = get_similar_startups(query)

    with st.spinner("🤔 Analyzing patterns from successful ventures..."):
        context_text = ""
        for i, (_, row) in enumerate(results.iterrows(), 1):
            context_text += f"{row['Project Name']}: {row['Problem Statement']} "

    prompt = f"""Generate exactly 3 distinct startup ideas for: {query}

For each idea, provide this information in order:
 Project Name
 Problem 
 Solution
 Target Audience
 Tech Stack
 Business Model
 Budget

Separate each idea clearly with '---' between them.

Context from similar ideas: {context_text}

Make sure each idea is well-formatted, readable, and practical."""

    st.subheader("✨ AI-Generated Startup Concepts:")

    with st.spinner("🧠 Generating innovative ideas..."):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
                stop=None,
                echo=False
            )
            
            raw_output = response['choices'][0]['text'].strip()
            cleaned_output = correct_text(raw_output)
            formatted_output = format_ai_output(cleaned_output)
            
            # Try splitting by '---', fallback to '💡 Project Name:'
            ideas = [idea.strip() for idea in formatted_output.split('---') if idea.strip()]
            if len(ideas) < 3:
                # Fallback: split by '💡 Project Name:'
                fallback_ideas = re.split(r'💡 Project Name:', formatted_output)
                ideas = ['💡 Project Name:' + i for i in fallback_ideas if i.strip()]
            
            st.success(f"✅ Generated {len(ideas)} AI Startup Ideas:")
            st.markdown("---")
            
            for i, idea_text in enumerate(ideas[:3], 1):
                if idea_text.strip():
                    clean_idea = idea_text.strip()
                    # Extract project name for expander header
                    match = re.search(r'💡 Project Name:\s*(.*)', clean_idea)
                    project_name = match.group(1).strip() if match else f"Idea   {i}"
                    
                    with st.expander(f"💡 {project_name}", expanded=True):
                        st.markdown(clean_idea.replace('\n', '  \n'))
                    st.markdown("---")
                        
        except Exception as e:
            st.error(f"⚡ Generation failed: {e}")
            st.info("Please try a different query or simpler request.")
