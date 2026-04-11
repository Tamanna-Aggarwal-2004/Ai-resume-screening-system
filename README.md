# 🤖 AI Resume Screening System

An intelligent resume screening system that uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze resumes and match them with job descriptions. The system helps recruiters identify the best candidates efficiently by calculating similarity scores and highlighting missing skills.

---

## 🚀 Live Demo

🔗 [View Live Demo]([YOUR_LIVE_LINK_HERE](https://ai-resume-screening-system-abcd.streamlit.app/))

---

## 🧠 Key Features

* 📄 Resume parsing and text extraction
* 🧠 NLP-based analysis of resumes
* 🎯 Job-description matching using ML
* 🔍 TF-IDF vectorization for feature extraction
* 🤝 Cosine similarity for accurate matching
* 📊 Match score generation (percentage)
* 📌 Highlights missing skills in resumes
* ⚡ Fast and real-time predictions

---

## 🛠️ Tech Stack

### Machine Learning & NLP

* Python
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

### Backend / Interface

* Streamlit (for interactive UI)
* FastAPI *(optional if used)*

### Data Processing

* Pandas
* NumPy

---

## ⚙️ How It Works

1. User uploads a resume and provides a job description
2. Text is cleaned and preprocessed using NLP techniques
3. TF-IDF converts text into numerical vectors
4. Cosine similarity is computed between resume and job description
5. System outputs:

   * Match score (%)
   * Missing skills
   * Recommendation insights

---

## 📂 Project Structure

```id="nqk9c2"
AI-Resume-Screening-System/
├── app.py                 # Streamlit UI  
├── main.py                # Backend logic (if FastAPI used)  
├── model.pkl              # Trained model  
├── tfidf.pkl              # TF-IDF vectorizer  
├── data/                  # Dataset  
├── utils/                 # Helper functions  
├── requirements.txt       # Dependencies  
└── README.md  
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash id="xmxapd"
git clone YOUR_GITHUB_REPO_LINK
cd AI-Resume-Screening-System
```

---

### 2️⃣ Install dependencies

```bash id="n9o3y1"
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

#### If using Streamlit

```bash id="ckl1m4"
streamlit run app.py
```

#### If using FastAPI

```bash id="d7k2xq"
uvicorn main:app --reload
```

---

## 📈 Future Enhancements

* 📊 Add ranking system for multiple resumes
* 🤖 Use deep learning models (BERT, transformers)
* 🌐 Deploy full-stack version
* 📂 Support multiple file formats (PDF, DOCX)
* 🧠 Improve skill extraction accuracy

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and submit pull requests.

---

## 📄 License

MIT License

---

## 📬 Contact

👩‍💻 **Tamanna Aggarwal**

🔗 GitHub: [GITHUB](https://github.com/Tamanna-Aggarwal-2004)
🔗 LinkedIn: [LINKEDIN](https://www.linkedin.com/in/tamanna-aggarwal-4a1102327/)
🌐 Portfolio: [PORTFOLIO](https://my-portfolio-zeta-cyan-gj6dgnmro9.vercel.app/)
---

✨ *Making hiring smarter with AI-powered resume analysis*
