## helper.py
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2 as pdf
import os
from pinecone import Pinecone, ServerlessSpec
import json
import re


def configure_genai(api_key):
    """Configure the Generative AI API."""
    genai.configure(api_key=api_key)


def calculate_similarity(jd, resume):
    """Calculate similarity between job description and resume using embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"),model="models/embedding-001")
    jd_embedding = embeddings.embed_query(jd)
    resume_embedding = embeddings.embed_query(resume)

    similarity = sum([a * b for a, b in zip(jd_embedding, resume_embedding)])
    return similarity

def calculate_similarity_pinecone(best_match, resume):
    """Calculate similarity between job description and resume using embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"),model="models/embedding-001")
    best_match = embeddings.embed_query(best_match)
    resume_embedding = embeddings.embed_query(resume)

    similarity = sum([a * b for a, b in zip(best_match, resume_embedding)])
    return similarity


def get_pinecone_matches(query):
    """Query Pinecone for matching resumes."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "resume-dataset-index"
    index=pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")))
    results = vector_store.similarity_search(query, k = 1)
    return[{
        "Category": result.metadata.get("Category"),
        "Content": result.page_content,
    } for result in results]


def extract_pdf_text(uploaded_file):
    """Extract text from PDF with enhanced error handling."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        if len(reader.pages) == 0:
            raise Exception("PDF file is empty")
        
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                
        if not text:
            raise Exception("No text could be extracted from the PDF")
            
        return " ".join(text)
        
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")



def prepare_prompt(resume_text, job_description, pinecone_result, cos_sim_jd_resume, cos_sim_pinecone_resume):
    """Prepare the input prompt with improved structure and validation."""
    if not resume_text or not job_description:
        raise ValueError("Resume text and job description cannot be empty")
        
    prompt_template = f"""
    Act as an expert ATS (Applicant Tracking System) specialist 
    Evaluate the following resume against the job description and top search result from the vector database. 
    Along with resume text, job description and search result also take into account the value of cosine similarity of job description and resume text along with cosine similarity of resume and search result to calculate the ATS score. Consider that the job market is highly competitive. Also provide detailed feedback for resume and suggest some improvements and missing keywords.
    Resume:
    {resume_text}
    
    Job Description: 
    {job_description}
    
    Top Search result based on the similarity search of resume in the Vector Database is : 
    {pinecone_result}
    
    The cosine similarity of resume of the user with job description is {cos_sim_jd_resume}.
    The cosine similarity of resume with that of the search result of vector database is {cos_sim_pinecone_resume}.
    Provide a response in the string format only
    """
    
    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip(),
        pinecone_result=pinecone_result.strip()
    )
    

def get_gemini_response(prompt):
    """Generate a response using Gemini with enhanced error handling and response validation."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Ensure response is not empty
        if not response or not response.text:
            raise Exception("Empty response received from Gemini")
            
    except:
        raise Exception("Problem with get_gemini_response")
    
    return response.text