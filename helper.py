from langchain_core.prompts import ChatPromptTemplate
import PyPDF2 as pdf

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

