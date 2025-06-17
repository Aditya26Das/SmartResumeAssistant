# 🎯 Smart Resume Assistant

**Smart Resume Assistant** is an AI-powered resume analyzer built with Streamlit. It helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) by evaluating job description compatibility, suggesting improvements, and identifying missing keywords — using embeddings, vector similarity, and LLM feedback.

---

## 🚀 Features

- 📄 Extracts and analyzes resume text (PDF)
- 🧠 Embedding-based similarity scoring with job descriptions
- 🔍 Vector-based matching using Pinecone
- ✨ Personalized AI feedback using Gemini Pro
- ✅ ATS scoring with keyword improvement suggestions

---

## 🛠️ Tech Stack

| Layer           | Tools/Tech Used                                |
|----------------|--------------------------------------------------|
| Frontend       | Streamlit                                       |
| NLP/AI         | Google Gemini Pro (via `google.generativeai`)   |
| Embeddings     | LangChain + Google Generative AI Embeddings     |
| Vector DB      | Pinecone                                         |
| PDF Parsing    | PyPDF2                                           |
| Utilities      | dotenv, Streamlit Extras                        |

---

## 📂 Folder Structure

SmartResumeAssistant/
├── app.py # Streamlit UI logic
├── helper.py # Core logic (embeddings, Pinecone, Gemini)
|-- notebooks
|      ├── Resume.ipynb # Jupyter notebook (exploratory / dev use)
├── .gitignore # Ignore venv, .env, pycache
├── requirements.txt # [You should add this file]
└── .env # [Create locally with API keys]



---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Aditya26Das/SmartResumeAssistant.git
cd SmartResumeAssistant
```
### 2. Set up the environment

```bash
python -m venv myenv
source myenv/bin/activate      # On Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables
#### Create a .env file in the root folder with the following:
```bash
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## How to Use ?

### 1. Run the app:
```bash
streamlit run app.py
```

### 2. Upload your resume (PDF)
### 3. Paste the job description
### 4. Click Analyze Resume
### 5. View ATS score, feedback, and keyword suggestions

# Author
## Aditya Das
## GitHub: [Aditya26Das](https://www.github.com/Aditya26Das)

# Future Improvements
- Dashboard for history of analysis
- Resume parser for structured formats
- Export feedback as PDF
- Upload multiple JDs for comparison
