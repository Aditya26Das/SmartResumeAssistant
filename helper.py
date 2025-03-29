from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2 as pdf
import os
from pinecone import Pinecone

def get_google_embeddings():
    """Initialize Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="models/embedding-001"
    )

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts using embeddings."""
    embeddings = get_google_embeddings()
    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    
    return sum(a * b for a, b in zip(emb1, emb2))

def get_pinecone_matches(query):
    """Query Pinecone for the most relevant match."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "resume-dataset-index"
    index = pc.Index(index_name)
    
    vector_store = PineconeVectorStore(
        index=index,
        embedding=get_google_embeddings()
    )
    
    results = vector_store.similarity_search(query, k=1)
    return [{
        "Category": result.metadata.get("Category"),
        "Content": result.page_content
    } for result in results]

def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        if len(reader.pages) == 0:
            raise Exception("PDF file is empty")
        
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text:
            raise Exception("No text could be extracted from the PDF")
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def prepare_prompt(resume_text, job_description, pinecone_result, cos_sim_jd_resume, cos_sim_pinecone_resume):
    """Prepare a structured prompt for evaluating the resume."""
    prompt = ChatPromptTemplate.from_template(
        """
        Act as an expert ATS (Applicant Tracking System) specialist. Evaluate the following resume against the job description and top search result from the vector database.
        Consider cosine similarity values but do not display them in the output. Provide an ATS score (percentage) and detailed feedback with suggested improvements and missing keywords.
        
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Top Search Result (from Vector Database):
        {pinecone_result}
        """
    )
    return prompt.format(resume_text=resume_text, job_description=job_description, pinecone_result=pinecone_result)

def get_gemini_response(prompt_text):
    """Generate response using Gemini AI via LangChain."""
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    chain = prompt_text | model | StrOutputParser()
    return chain.invoke({})
