import streamlit as st
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from helper import extract_pdf_text, prepare_prompt
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

def init_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def main():
    load_dotenv()
    init_session_state()

    google_api_key = os.getenv("GOOGLE_API_KEY") 
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not google_api_key or not pinecone_api_key:
        st.error("Please set the API keys in your .env file")
        return

    with st.sidebar:
        st.title("🎯 Smart ATS")
        st.subheader("About")
        st.write("""
        This smart ATS helps you:
        - Evaluate resume-job description match
        - Identify missing keywords
        - Get personalized improvement suggestions
        """)

    st.title("🎯 Smart ATS Resume Analyzer")
    st.subheader("Optimize Your Resume for ATS")

    jd = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        help="Enter the complete job description for accurate analysis"
    )

    uploaded_file = st.file_uploader(
        "Resume (PDF)",
        type="pdf",
        help="Upload your resume in PDF format"
    )

    if st.button("Analyze Resume", disabled=st.session_state.processing):
        if not jd:
            st.warning("Please provide a job description.")
            return
        if not uploaded_file:
            st.warning("Please upload a resume in PDF format.")
            return

        st.session_state.processing = True

        try:
            with st.spinner("🎯 Analyzing your resume..."):
                resume_text = extract_pdf_text(uploaded_file)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": False}
                )

                jd_embedding = embeddings.embed_query(jd)
                resume_embedding = embeddings.embed_query(resume_text)
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index("resume-dataset-index")
                vector_store = PineconeVectorStore(index=index, embedding=embeddings)
                results = vector_store.similarity_search(jd, k=1)
                pinecone_results = results[0].page_content if results else ""
                similarity = sum(a * b for a, b in zip(jd_embedding, resume_embedding))

                pinecone_embedding = embeddings.embed_query(pinecone_results)
                pinecone_similarity = sum(a * b for a, b in zip(pinecone_embedding, resume_embedding))

                input_prompt = prepare_prompt(
                    resume_text=resume_text,
                    job_description=jd,
                    pinecone_result=pinecone_results,
                    cos_sim_pinecone_resume=pinecone_similarity,
                    cos_sim_jd_resume=similarity
                )
                model = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=google_api_key)

                prompt = ChatPromptTemplate.from_template(input_prompt)
                response = model.invoke(prompt.format())

                st.success("🎯 Analysis Complete!")
                st.title("Resume Feedback : ")
                st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()
