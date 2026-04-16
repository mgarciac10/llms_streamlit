import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import tiktoken
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import time

# Page Configuration
st.set_page_config(page_title="LLM Deconstruction Workshop", layout="wide")
st.title("🛠️ Workshop: Dismantling LLMs")
st.sidebar.markdown("### Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# --- NAVIGATION ---
tabs = st.tabs([
    "1. Tokenization Lab", 
    "2. Word Embeddings", 
    "3. Groq Inference", 
    "4. Performance Metrics"
])

# --- MODULE 1: TOKENIZER LABORATORY ---
with tabs[0]:
    st.header("1. The Tokenizer Lab")
    st.info("Converting text into numerical IDs.") # [cite: 9]
    
    text_input = st.text_area("Enter text to tokenize:", "Learning transformers is fascinating!")
    
    if text_input:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text_input)
        token_strings = [encoding.decode([t]) for t in tokens]
        
        # Display colored tokens [cite: 23]
        st.subheader("Colored Tokens")
        html_tokens = ""
        colors = ["#FFDDC1", "#C1E1FF"]
        for i, t in enumerate(token_strings):
            color = colors[i % 2]
            html_tokens += f'<span style="background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px; color: black;">{t}</span>'
        st.markdown(html_tokens, unsafe_allow_html=True)
        
        # Token ID Mapping [cite: 24]
        st.subheader("Token ID Mapping")
        df_tokens = pd.DataFrame({"Token": token_strings, "ID": tokens})
        st.table(df_tokens)
        
        # Comparative Metrics [cite: 25]
        col1, col2 = st.columns(2)
        col1.metric("Number of Characters", len(text_input))
        col2.metric("Number of Tokens", len(tokens))

# --- MODULE 2: WORD EMBEDDINGS GEOMETRY ---
with tabs[1]:
    st.header("2. Geometry of Words")
    st.write("Verifying semantic relationships in embedding space.") # [cite: 27]
    
    # Standard example words [cite: 28, 32]
    word_list_input = st.text_input("Enter words (comma separated):", "king, man, woman, queen, apple, fruit")
    words = [w.strip() for w in word_list_input.split(",")]

    if len(words) >= 2:
        # Using a local transformer for embeddings [cite: 29]
        @st.cache_resource
        def load_embed_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        model = load_embed_model()
        embeddings = model.encode(words)
        
        # Dimensionality Reduction (PCA) [cite: 30]
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(embeddings)
        
        df_embeddings = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
        df_embeddings['word'] = words
        
        # Interactive Plotly Chart [cite: 31]
        fig = px.scatter(df_embeddings, x='x', y='y', text='word', title="2D Projection of Word Embeddings")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        
        st.latex(r"(king) - (man) + (woman) \approx (queen)") # [cite: 32]
    else:
        st.warning("Please enter at least 2 words.")

# --- MODULE 3 & 4: INFERENCE & PERFORMANCE ---
with tabs[2] or tabs[3]:
    if not api_key:
        st.error("Please provide a Groq API Key in the sidebar to use Modules 3 and 4.")
    else:
        client = Groq(api_key=api_key)
        
        st.header("3. Inference & Reasoning")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            sys_prompt = st.text_area("System Prompt", "You are a helpful and concise assistant.") # [cite: 36]
            user_prompt = st.text_area("User Prompt", "Explain why the sky is blue.")
        
        with col_p2:
            temp = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1) # [cite: 34]
            top_p = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9) # [cite: 35]
            st.caption("Lower temp (<0.3) = Deterministic; Higher temp (>0.7) = Creative.") # [cite: 34]

        if st.button("Generate Response"):
            start_time = time.time()
            completion = client.chat.completions.create(
                model="llama3-8b-8192", # Low cost/param model 
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                top_p=top_p
            )
            end_time = time.time()
            
            response_text = completion.choices[0].message.content
            usage = completion.usage # Metadata from Groq API [cite: 38]

            # Displaying Output
            with tabs[2]:
                st.subheader("Model Response")
                st.write(response_text)
            
            # Displaying Metrics [cite: 37, 39, 40, 41]
            with tabs[3]:
                st.header("4. Performance Metrics")
                m1, m2, m3 = st.columns(3)
                
                # Groq provides specific usage stats
                m1.metric("Time per Token (ms)", f"{ (usage.completion_time * 1000) / usage.completion_tokens :.2f}")
                m2.metric("Throughput (tokens/s)", f"{usage.completion_tokens / usage.completion_time :.2f}")
                m3.metric("Total Tokens", usage.total_tokens)
                
                st.write(f"**Input Tokens:** {usage.prompt_tokens}")
                st.write(f"**Output Tokens:** {usage.completion_tokens}")
