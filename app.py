import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# -----------------------------
# Page config FIRST (important)
# -----------------------------
st.set_page_config(page_title="Resume Job Match Scorer",page_icon="📄",layout="wide")

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!  
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Measures how your resume matches a job description
    - Identify important job keywords
    - Improve your reseume based on missing terms
    """)
    st.header("How It works")
    st.write("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4. Review score & suggetion
    """)

# -----------------------------
# NLTK setup (FIXED)
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')


# -----------------------------
# Cache model (VERY IMPORTANT 🔥)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🚀 AI Resume Job Match Analyzer")
st.write("Upload your resume and compare with any job description")

# -----------------------------
# PDF reader
# -----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# -----------------------------
# Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# Keyword extraction (AUTO)
# -----------------------------
def extract_keywords(text, top_n=20):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    freq = Counter(words)
    return set([word for word, _ in freq.most_common(top_n)])


# -----------------------------
# TF-IDF similarity
# -----------------------------
def tfidf_similarity(resume, jd):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform([resume, jd])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100


# -----------------------------
# Semantic similarity
# -----------------------------
def semantic_similarity(resume, jd):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)) * 100


# -----------------------------
# Keyword overlap
# -----------------------------
def keyword_score(resume, jd):
    r_keys = extract_keywords(resume)
    j_keys = extract_keywords(jd)

    if len(j_keys) == 0:
        return 0, [], []

    overlap = r_keys.intersection(j_keys)
    missing = j_keys - r_keys

    score = (len(overlap) / len(j_keys)) * 100
    return score, list(overlap), list(missing)


# -----------------------------
# Final score
# -----------------------------
def final_score(resume, jd):
    resume_clean = clean_text(resume)
    jd_clean = clean_text(jd)

    tfidf = tfidf_similarity(resume_clean, jd_clean)
    semantic = semantic_similarity(resume, jd)
    keyword_s, common, missing = keyword_score(resume_clean, jd_clean)

    score = (0.5 * semantic) + (0.3 * tfidf) + (0.2 * keyword_s)

    return round(score, 2), common[:10], missing[:10]


# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description", height=200)

if st.button("🔍 Analyze Match"):
    if not uploaded_file or not job_description:
        st.warning("Please upload resume and paste job description")
    else:
        with st.spinner("Analyzing..."):

            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                st.error("Could not extract text from PDF")
            else:
                score, common, missing = final_score(resume_text, job_description)

                # -----------------------------
                # Output
                # -----------------------------
                # -----------------------------
# Match Score UI (Premium)
# -----------------------------
                st.metric("Score", f"{score:.2f}%")

# Color based on score
                if score < 40:
                    color = "#ff4b4b"
                elif score < 70:
                    color = "#ffa726"
                else:
                    color = "#00c853"

                # Gradient bar (same CSS, no text)
                st.markdown(f"""
                <div style="
                    background:#eee;
                    border-radius:12px;
                    height:40px;
                    width:100%;
                    margin-bottom: 20px;
                ">
                    <div style="
                        width:{score}%;
                        background:linear-gradient(90deg,{color},#00c6ff);
                        height:40px;
                        border-top-left-radius:12px;
                        border-bottom-left-radius:12px;
                        text-align:center;
                        color:white;
                        font-size:18px;
                        line-height:20px;
                        margin-bottom: 20px;
                    ">
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # feedback
                if score < 40:
                    st.error("Low Match ❌ Improve resume")
                elif score < 70:
                    st.warning("Moderate Match ⚠️")
                else:
                    st.success("Strong Match ✅")

                # keywords
                st.subheader("✅ Matching Keywords")
                st.write(common if common else "None")

                st.subheader("❌ Missing Keywords")
                st.write(missing if missing else "None")
