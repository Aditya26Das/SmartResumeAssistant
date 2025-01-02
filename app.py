## app.py
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import json
from dotenv import load_dotenv
from helper import (
    configure_genai,
    extract_pdf_text,
    prepare_prompt,
    calculate_similarity_pinecone,
    calculate_similarity,
    get_pinecone_matches,
    get_gemini_response
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
    
    # Configure Generative AI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY in your .env file")
        return
        
    try:
        configure_genai(api_key)
    except Exception as e:
        st.error(f"Failed to configure API: {str(e)}")
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
                print(f"Data type of resume_text : {type(resume_text)}")
                
                # Query Pinecone for most matched resume
                pinecone_results = get_pinecone_matches(jd)[0]
                pinecone_results = pinecone_results['Content']
                print(pinecone_results)
                print(f"Data type of pinecone_results : {type(pinecone_results)}")
                
                # Prepare embeddings
                similarity = calculate_similarity(jd, resume_text)
                pinecone_similarity = calculate_similarity_pinecone(pinecone_results,resume_text)
                
                # Calculate ATS score
                ats_score = ((similarity + pinecone_similarity) / 2) * 100
                
                # Geimin
                input_prompt = prepare_prompt(resume_text=resume_text,job_description=jd)
                
                # Display results
                st.success("🎯 Analysis Complete!")
                
                # Match percentage
                st.metric("ATS Score", f"{ats_score:.2f}%")
                
                # Missing keywords
                st.title("Resume Feedback : ")
                res=get_gemini_response(input_prompt)
                st.write(res)
                                

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()