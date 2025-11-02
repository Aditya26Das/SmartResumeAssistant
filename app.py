import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from helper import (
    extract_pdf_text,
    prepare_prompt
)

def init_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    init_session_state()
    
    # Configure API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not google_api_key or not pinecone_api_key:
        st.error("Please set the API keys in your .env file")
        return

    # Sidebar
    with st.sidebar:
        st.title("🎯 Smart ATS")
        st.subheader("About")
        st.write("""
        This smart ATS helps you:
        - Evaluate resume-job description match
        - Identify missing keywords
        - Get personalized improvement suggestions
        """)

    # Main content
    st.title("🎯 Smart ATS Resume Analyzer")
    st.subheader("Optimize Your Resume for ATS")
    
    # Input sections with validation
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

    # Process button with loading state
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
                # Extract text from PDF
                resume_text = extract_pdf_text(uploaded_file)
                
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/gemini-embedding-001")
                jd_embedding = embeddings.embed_query(jd)
                resume_embedding = embeddings.embed_query(resume_text)
                
                # Query Pinecone for most matched resume
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index("resume-dataset-index")
                vector_store = PineconeVectorStore(index=index, embedding=embeddings)
                results = vector_store.similarity_search(jd, k=1)
                pinecone_results = results[0].page_content if results else ""
                
                # Calculate similarity scores
                model = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
                similarity = sum([a * b for a, b in zip(jd_embedding, resume_embedding)])
                pinecone_embedding = embeddings.embed_query(pinecone_results)
                pinecone_similarity = sum([a * b for a, b in zip(pinecone_embedding, resume_embedding)])
                
                # Prepare input prompt
                input_prompt = prepare_prompt(
                    resume_text=resume_text,
                    job_description=jd,
                    pinecone_result=pinecone_results,
                    cos_sim_pinecone_resume=pinecone_similarity,
                    cos_sim_jd_resume=similarity
                )
                
                # Generate response using LangChain's chat model
                prompt = ChatPromptTemplate.from_template(input_prompt)
                response = model.invoke(prompt.format())
                
                # Display results
                st.success("🎯 Analysis Complete!")
                st.title("Resume Feedback : ")
                st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()
