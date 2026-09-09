import streamlit as st
import pandas as pd
from pdf_loader import load_pdfs, chunk_text
from embedder import Embedder
from vector_store import VectorStore
from qa_engine import generate_answer, generate_mcqs
from utils import save_chat_history, format_sources
from translator import translate_text
import concurrent.futures

# -------------------- Page Setup --------------------
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for dark grey professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 25%, #4a5568 50%, #2d3748 75%, #1a202c 100%);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(74, 85, 104, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: #f7fafc;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(226, 232, 240, 0.05) 50%, transparent 70%);
        animation: shimmer 4s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .chat-container {
        background: rgba(45, 55, 72, 0.8);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #718096;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .answer-box {
        background: rgba(26, 32, 44, 0.9);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        padding: 2rem;
        border-radius: 20px;
        border-left: 5px solid #68d391;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .answer-box h4 {
        color: #a0aec0;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .answer-box p {
        color: #cbd5e0;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .question-box {
        background: linear-gradient(135deg, rgba(74, 85, 104, 0.8) 0%, rgba(113, 128, 150, 0.8) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(160, 174, 192, 0.2);
        padding: 1.2rem;
        border-radius: 15px;
        border-left: 4px solid #90cdf4;
        margin: 0.5rem 0;
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .sidebar-section {
        background: rgba(26, 32, 44, 0.9);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
    }
    
    .sidebar-section h4 {
        color: #cbd5e0;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .sidebar-section p {
        color: #a0aec0;
        font-weight: 400;
    }
    
    .success-banner {
        background: linear-gradient(135deg, rgba(56, 178, 172, 0.2) 0%, rgba(72, 187, 120, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(72, 187, 120, 0.3);
        color: #9ae6b4;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        font-weight: 500;
    }
    
    .quiz-card {
        background: rgba(26, 32, 44, 0.9);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border-left: 5px solid #fc8181;
        color: #e2e8f0;
    }
    
    .quiz-card h4 {
        color: #fed7d7;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .quiz-card p {
        color: #cbd5e0;
        font-size: 1.1rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .metric-card {
        background: rgba(26, 32, 44, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .metric-card h3 {
        color: #cbd5e0;
        font-weight: 700;
    }
    
    .metric-card h4 {
        color: #a0aec0;
        font-size: 1.5rem;
    }
    
    .metric-card p {
        color: #718096;
        font-weight: 500;
    }
    
    /* Enhanced input styling for dark theme */
    .stTextInput > div > div > input {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 2px solid rgba(113, 128, 150, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #718096 !important;
        box-shadow: 0 0 0 3px rgba(113, 128, 150, 0.2) !important;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 2px solid rgba(113, 128, 150, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #718096 !important;
        box-shadow: 0 0 0 3px rgba(113, 128, 150, 0.2) !important;
    }
    
    .stSelectbox > div > div > select {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 2px solid rgba(113, 128, 150, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 2px solid rgba(113, 128, 150, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    /* Dark theme button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%) !important;
        border: 1px solid rgba(113, 128, 150, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #718096 0%, #4a5568 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
        border-color: rgba(160, 174, 192, 0.4) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #718096 0%, #4a5568 100%) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(160, 174, 192, 0.3) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #a0aec0 0%, #718096 100%) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        border-color: rgba(203, 213, 224, 0.4) !important;
    }
    
    /* Dark theme radio button styling */
    .stRadio > div {
        background: rgba(45, 55, 72, 0.8);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(113, 128, 150, 0.2);
    }
    
    .stRadio > div > label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Dark theme expander styling */
    .streamlit-expanderHeader {
        background: rgba(45, 55, 72, 0.9) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(113, 128, 150, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(26, 32, 44, 0.8) !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(113, 128, 150, 0.2) !important;
        border-top: none !important;
        color: #e2e8f0 !important;
    }
    
    /* Dark theme progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #718096 0%, #4a5568 100%) !important;
    }
    
    /* Dark theme file uploader */
    .stFileUploader > div {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 2px dashed rgba(113, 128, 150, 0.4) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        color: #e2e8f0 !important;
    }
    
    /* Dark theme dataframe styling */
    .stDataFrame {
        background: rgba(26, 32, 44, 0.95) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(113, 128, 150, 0.2) !important;
        color: #e2e8f0 !important;
    }
    
    /* Dark theme alert boxes */
    .stAlert {
        background: rgba(45, 55, 72, 0.9) !important;
        border-radius: 12px !important;
        border-left: 4px solid #718096 !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Dark theme section headers */
    h1, h2, h3, h4, h5, h6 {
        color: #cbd5e0 !important;
        font-weight: 600 !important;
    }
    
    /* Dark theme labels and text */
    label {
        color: #a0aec0 !important;
        font-weight: 500 !important;
    }
    
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Custom dark glass effect for containers */
    .glass-container {
        background: rgba(26, 32, 44, 0.3);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        color: #e2e8f0;
    }
    
    /* Dark theme info/success/error messages */
    .stInfo {
        background: rgba(56, 178, 172, 0.2) !important;
        color: #81e6d9 !important;
        border-color: rgba(56, 178, 172, 0.4) !important;
    }
    
    .stSuccess {
        background: rgba(72, 187, 120, 0.2) !important;
        color: #9ae6b4 !important;
        border-color: rgba(72, 187, 120, 0.4) !important;
    }
    
    .stError {
        background: rgba(252, 129, 129, 0.2) !important;
        color: #feb2b2 !important;
        border-color: rgba(252, 129, 129, 0.4) !important;
    }
    
    .stWarning {
        background: rgba(237, 137, 54, 0.2) !important;
        color: #f6ad55 !important;
        border-color: rgba(237, 137, 54, 0.4) !important;
    }
    
    /* Spinner color adjustment */
    .stSpinner {
        color: #a0aec0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="main-header">
    <h1>📚 StudyMate - AI Academic Assistant</h1>
    <p>Your intelligent companion for academic research and learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Session State --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "mcqs" not in st.session_state:
    st.session_state.mcqs = []
if "show_next" not in st.session_state:
    st.session_state.show_next = False

# -------------------- Language Selection --------------------
languages = {
    "english": "en",
    "afrikaans": "af", "albanian": "sq", "amharic": "am", "arabic": "ar",
    "armenian": "hy", "assamese": "as", "aymara": "ay", "azerbaijani": "az",
    "bambara": "bm", "basque": "eu", "belarusian": "be", "bengali": "bn",
    "bhojpuri": "bho", "bosnian": "bs", "bulgarian": "bg", "catalan": "ca",
    "cebuano": "ceb", "chichewa": "ny", "chinese (simplified)": "zh-CN",
    "chinese (traditional)": "zh-TW", "corsican": "co", "croatian": "hr",
    "czech": "cs", "danish": "da", "dhivehi": "dv", "dogri": "doi", "dutch": "nl",
    "esperanto": "eo", "estonian": "et", "ewe": "ee",
    "filipino": "tl", "finnish": "fi", "french": "fr", "frisian": "fy",
    "galician": "gl", "georgian": "ka", "german": "de", "greek": "el",
    "guarani": "gn", "gujarati": "gu", "haitian creole": "ht", "hausa": "ha",
    "hawaiian": "haw", "hebrew": "iw", "hindi": "hi", "hmong": "hmn",
    "hungarian": "hu", "icelandic": "is", "igbo": "ig", "ilocano": "ilo",
    "indonesian": "id", "irish": "ga", "italian": "it", "japanese": "ja",
    "javanese": "jw", "kannada": "kn", "kazakh": "kk", "khmer": "km",
    "kinyarwanda": "rw", "konkani": "gom", "korean": "ko", "krio": "kri",
    "kurdish (kurmanji)": "ku", "kurdish (sorani)": "ckb", "kyrgyz": "ky",
    "lao": "lo", "latin": "la", "latvian": "lv", "lingala": "ln",
    "lithuanian": "lt", "luganda": "lg", "luxembourgish": "lb", "macedonian": "mk",
    "maithili": "mai", "malagasy": "mg", "malay": "ms", "malayalam": "ml",
    "maltese": "mt", "maori": "mi", "marathi": "mr", "meiteilon (manipuri)": "mni-Mtei",
    "mizo": "lus", "mongolian": "mn", "myanmar": "my", "nepali": "ne",
    "norwegian": "no", "odia (oriya)": "or", "oromo": "om", "pashto": "ps",
    "persian": "fa", "polish": "pl", "portuguese": "pt", "punjabi": "pa",
    "quechua": "qu", "romanian": "ro", "russian": "ru", "samoan": "sm",
    "sanskrit": "sa", "scots gaelic": "gd", "sepedi": "nso", "serbian": "sr",
    "sesotho": "st", "shona": "sn", "sindhi": "sd", "sinhala": "si",
    "slovak": "sk", "slovenian": "sl", "somali": "so", "spanish": "es",
    "sundanese": "su", "swahili": "sw", "swedish": "sv", "tajik": "tg",
    "tamil": "ta", "tatar": "tt", "telugu": "te", "thai": "th", "tigrinya": "ti",
    "tsonga": "ts", "turkish": "tr", "turkmen": "tk", "twi": "ak",
    "ukrainian": "uk", "urdu": "ur", "uyghur": "ug", "uzbek": "uz",
    "vietnamese": "vi", "welsh": "cy", "xhosa": "xh", "yiddish": "yi",
    "yoruba": "yo", "zulu": "zu"
}

# -------------------- Main Layout --------------------
col1, col2 = st.columns([2, 1])

with col1:
    # -------------------- Language Selection --------------------
    st.markdown("### 🌍 Language Settings")
    target_lang = st.selectbox("Select output language", list(languages.keys()), key="main_lang")
    
    # -------------------- Question Input --------------------
    st.markdown("### 💬 Ask Your Question")
    question = st.text_area(
        "Enter your question about the uploaded documents:", 
        height=100,
        placeholder="What would you like to know about your study materials?",
        help="Type your question here and click 'Get Answer' to receive detailed information from your documents."
    )
    
    # -------------------- Get Answer --------------------
    col_ask1, col_ask2 = st.columns([1, 1])
    
    with col_ask1:
        get_answer_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    
    with col_ask2:
        clear_chat_btn = st.button("🗑 Clear Chat", use_container_width=True)
        if clear_chat_btn:
            st.session_state.chat_history = []
            st.experimental_rerun()

    if get_answer_btn and question and st.session_state.vector_store:
        with st.spinner("🤖 Generating answer..."):
            embedder = Embedder()
            query_embedding = embedder.embed([question])[0]

            # Retrieve top 2 relevant chunks for faster generation
            sources = st.session_state.vector_store.search(query_embedding, top_k=2)
            context = " ".join([src for src, _ in sources])

            answer = generate_answer(context, question, target_lang=languages[target_lang])

            # Save chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })
            
            st.experimental_rerun()

    # -------------------- Current Answer Display --------------------
    if st.session_state.chat_history:
        latest_qa = st.session_state.chat_history[-1]
        
        st.markdown("### 💡 Latest Answer")
        st.markdown(f"""
        <div class="answer-box">
            <h4>Question:</h4>
            <p>{latest_qa['question']}</p>
            <h4>Answer ({target_lang}):</h4>
            <p>{latest_qa['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------- Chat History (Collapsible) --------------------
    if st.session_state.chat_history:
        st.markdown("### 📝 Chat History")
        
        for idx, entry in enumerate(st.session_state.chat_history, start=1):
            # Each question has its own expander (no nested expander)
            with st.expander(f"Q{idx}: {entry['question']}", expanded=False):
                st.markdown(f"""
                <div class="chat-container">
                    <strong>Answer:</strong> {entry['answer']}
                </div>
                """, unsafe_allow_html=True)

                # Display sources directly
                st.markdown("Sources:")
                st.markdown(format_sources(entry["sources"]))

        # Download chat history button
        if st.button("💾 Download Chat History", use_container_width=True):
            save_chat_history(st.session_state.chat_history)
            st.success("💾 Chat history saved as chat_history.txt")
    
    # -------------------- Quiz Section --------------------
    st.markdown("### 📝 Interactive Quiz Generator")
    
    quiz_languages = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", 
        "French": "fr", "Spanish": "es"
    }
    
    with st.container():
        col_quiz1, col_quiz2 = st.columns([2, 1])
        
        with col_quiz1:
            quiz_topic = st.text_input("📖 Enter quiz topic from your PDFs:", 
                                     placeholder="e.g., Machine Learning, History, Biology",
                                     help="Enter a topic that exists in your uploaded documents")
        
        with col_quiz2:
            num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
        
        col_lang, col_gen = st.columns([1, 1])
        
        with col_lang:
            target_lang_name = st.selectbox("Quiz Language", options=list(quiz_languages.keys()))
        
        with col_gen:
            st.write("")  # Spacing
            generate_quiz_btn = st.button("🎯 Generate Quiz", type="primary", use_container_width=True)

    # Cached function for quiz generation
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_generate(context, topic, num_q, target_code):
        return generate_mcqs(context, topic, num_questions, target_code)

    # Generate MCQs
    if generate_quiz_btn and quiz_topic and st.session_state.vector_store:
        with st.spinner("🤖 Generating quiz questions..."):
            embedder = Embedder()
            query_embedding = embedder.embed([quiz_topic])[0]
            
            sources = st.session_state.vector_store.search(query_embedding, top_k=2)
            context = " ".join([src for src, _ in sources])[:2000]
            
            target_code = quiz_languages.get(target_lang_name, "en")
            st.session_state.mcqs = cached_generate(context, quiz_topic, num_questions, target_code)[:num_questions]
            st.session_state.current_q = 0
            st.session_state.user_answers = []
            st.session_state.show_next = False

    # Quiz Display
    if st.session_state.mcqs:
        mcqs = st.session_state.mcqs
        cur = st.session_state.current_q

        if cur >= len(mcqs):
            # Quiz completed - show results
            results = []
            for i, q in enumerate(mcqs):
                user_sel = st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else ""
                correct = q["answer"]
                correct_flag = (user_sel == correct)
                results.append({
                    "Q#": i+1,
                    "Question": q["question"],
                    "Selected": user_sel,
                    "Correct Answer": correct,
                    "Is Correct": "✅ Yes" if correct_flag else "❌ No"
                })

            df = pd.DataFrame(results)
            score = df["Is Correct"].str.contains("Yes").sum()
            
            st.balloons()
            st.markdown(f"""
            <div class="success-banner">
                <h3>🎉 Quiz Completed!</h3>
                <h2>Your Score: {score}/{len(mcqs)} ({score/len(mcqs)*100:.1f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display results table
            st.dataframe(df, use_container_width=True)
            
            col_download, col_restart = st.columns([1, 1])
            
            with col_download:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_bytes,
                    file_name="quiz_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_restart:
                if st.button("🔄 Start New Quiz", use_container_width=True):
                    st.session_state.current_q = 0
                    st.session_state.user_answers = []
                    st.session_state.mcqs = []
                    st.experimental_rerun()

        else:
            # Display current question
            q = mcqs[cur]
            
            st.markdown(f"""
            <div class="quiz-card">
                <h4>Question {cur+1} of {len(mcqs)}</h4>
                <p style="font-size: 1.1em; font-weight: 500;">{q['question']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress = (cur + 1) / len(mcqs)
            st.progress(progress)
            
            selected = st.radio("Select your answer:", q["options"], key=f"q{cur}")
            
            col_submit, col_progress = st.columns([1, 2])
            
            with col_submit:
                submit_btn = st.button("✅ Submit Answer", key=f"submit_{cur}", use_container_width=True)
            
            with col_progress:
                st.metric("Progress", f"{cur+1}/{len(mcqs)}", f"{progress:.1%}")

            if submit_btn:
                st.session_state.user_answers.append(selected)
                correct = q["answer"]
                
                if selected == correct:
                    st.success("✅ Correct! Well done!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {correct}")
                
                st.session_state.show_next = True

            if st.session_state.show_next and cur < len(mcqs) - 1:
                if st.button("➡ Next Question", key=f"next_{cur}", type="primary"):
                    st.session_state.current_q += 1
                    st.session_state.show_next = False
                    st.experimental_rerun()
            elif st.session_state.show_next and cur == len(mcqs) - 1:
                if st.button("🏁 Finish Quiz", key=f"finish_{cur}", type="primary"):
                    st.session_state.current_q += 1
                    st.session_state.show_next = False
                    st.experimental_rerun()

# -------------------- Sidebar (PDF Processing) --------------------
with col2:
    st.markdown("### 📂 Document Management")
    
    # File upload section
    st.markdown("""
    <div class="sidebar-section">
        <h4>📁 Upload Study Materials</h4>
        <p>Upload one or more PDF documents to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your study materials, research papers, or textbooks"
    )
    
    # Processing section
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")
        
        # Display file names
        with st.expander("📋 Selected Files", expanded=True):
            for file in uploaded_files:
                st.write(f"• {file.name}")
        
        # Process button
        process_btn = st.button("⚡ Process PDFs", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("⏳ Reading and processing PDFs..."):
                texts = []
                progress_bar = st.progress(0)
                
                for i, pdf_text in enumerate(load_pdfs(uploaded_files)):
                    chunks = chunk_text(pdf_text)
                    texts.extend(chunks)
                    progress_bar.progress((i + 1) / len(uploaded_files))

                embedder = Embedder()
                embeddings = embedder.embed(texts)

                store = VectorStore(dim=len(embeddings[0]))
                store.add(embeddings, texts)
                st.session_state.vector_store = store

                st.success("✅ PDFs processed successfully!")
                st.info(f"📊 Processed {len(texts)} text chunks")
    else:
        st.markdown("""
        <div class="sidebar-section">
            <p>👆 Please upload PDF files to begin using StudyMate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Status section
    st.markdown("### 📊 System Status")
    
    # Status metrics
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        st.markdown("""
        <div class="metric-card">
            <h4>📚</h4>
            <p>Documents</p>
            <h3>{}</h3>
        </div>
        """.format(len(uploaded_files) if uploaded_files else 0), unsafe_allow_html=True)
    
    with col_status2:
        st.markdown("""
        <div class="metric-card">
            <h4>💬</h4>
            <p>Q&A History</p>
            <h3>{}</h3>
        </div>
        """.format(len(st.session_state.chat_history)), unsafe_allow_html=True)
    
    # System status
    vector_status = "🟢 Ready" if st.session_state.vector_store else "🔴 Not Ready"
    st.markdown(f"Vector Store: {vector_status}")
    
    # Additional Features Section
    st.markdown("### ⚡ Quick Actions")
    
    # Quick stats in glass containers
    if st.session_state.chat_history:
        avg_response_length = sum(len(qa['answer']) for qa in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.markdown(f"""
        <div class="glass-container">
            <h5>📈 Session Statistics</h5>
            <p><strong>Total Questions:</strong> {len(st.session_state.chat_history)}</p>
            <p><strong>Avg Response Length:</strong> {avg_response_length:.0f} characters</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("""
    <div class="sidebar-section">
        <h4>✨ Features</h4>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>🌐</span>
                <span>Multi-language Support</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>🧠</span>
                <span>AI-Powered Q&A</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>📝</span>
                <span>Interactive Quizzes</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>📚</span>
                <span>Source Citations</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>💾</span>
                <span>Export Results</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Help section
    with st.expander("💡 How to Use StudyMate", expanded=False):
        st.markdown("""
        Getting Started:
        1. 📤 Upload your PDF study materials
        2. ⚡ Click "Process PDFs" to index the content
        3. 💬 Ask questions about your documents
        4. 🎯 Generate interactive quizzes for self-assessment
        
        Pro Tips:
        - Use specific questions for better answers
        - Try different languages for output
        - Generate quizzes to test your knowledge
        - Download your chat history for later review
        
        Supported Features:
        - 🌍 100+ languages supported
        - 📊 Real-time progress tracking
        - 🎨 Professional dark interface
        - 🔍 Intelligent document search
        """)
    
    # Theme toggle (placeholder for future implementation)
    st.markdown("### 🎨 Appearance")
    st.markdown("""
    <div class="glass-container">
        <p style="text-align: center; color: #a0aec0;">
            Dark professional theme active
        </p>
    </div>
    """, unsafe_allow_html=True)
