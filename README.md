# 🤖 AI Resume Screening System

An intelligent resume screening system that uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze resumes and match them with job descriptions. The system helps recruiters identify the best candidates efficiently by calculating similarity scores and highlighting missing skills.

---

## 🚀 Live Demo

🔗 [View Live Demo](https://ai-resume-screening-system-abcd.streamlit.app/)

---

## 🧠 Key Features

* 📄 Resume parsing and text extraction
* 🧠 NLP-based analysis of resumes
* 🎯 Job-description matching using Machine Learning
* 🔍 TF-IDF vectorization for feature extraction
* 🤝 Cosine similarity for accurate matching
* 📊 Match score generation (percentage)
* 📌 Highlights missing skills in resumes
* ⚡ Fast and real-time predictions using Streamlit

---

## 🛠️ Tech Stack

### Machine Learning & NLP

* Python
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

### Interface

* Streamlit

### Data Processing

* Pandas
* NumPy

---

## ⚙️ How It Works

1. User uploads a resume and enters a job description
2. Text is cleaned and preprocessed using NLP techniques
3. TF-IDF converts text into numerical vectors
4. Cosine similarity is computed between resume and job description
5. System outputs:

   * Match score (%)
   * Missing skills
   * Recommendation insights

---

## 📂 Project Structure

```bash
AI-Resume-Screening-System/
├── app.py              # Streamlit application  
├── model.pkl           # Trained ML model  
├── tfidf.pkl           # TF-IDF vectorizer  
├── data/               # Dataset  
├── utils/              # Helper functions  
├── requirements.txt    # Dependencies  
└── README.md  
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Tamanna-Aggarwal-2004/AI-Resume-Screening-System
cd AI-Resume-Screening-System
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

* 📊 Ranking system for multiple resumes
* 🤖 Advanced NLP models (BERT, Transformers)
* 📂 Support for PDF and DOCX parsing
* 🧠 Improved skill extraction

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and submit pull requests.

---

## 📄 License

MIT License

---

## 📬 Contact

👩‍💻 **Tamanna Aggarwal**

🔗 GitHub: https://github.com/Tamanna-Aggarwal-2004
🔗 LinkedIn: https://www.linkedin.com/in/tamanna-aggarwal-4a1102327/
🌐 Portfolio: https://my-portfolio-zeta-cyan-gj6dgnmro9.vercel.app/

---

✨ *Making hiring smarter with AI-powered resume analysis*
