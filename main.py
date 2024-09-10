from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from PyPDF2 import PdfReader
from gtts import gTTS
import os
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Initialize Gradio Client for LLM with retry mechanism
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(httpx.RequestError))
def initialize_gradio_client():
    return Client("rohitmenonhart/mistral-super-f2")

client = initialize_gradio_client()

# Function to get response from LLM
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(httpx.RequestError))
def get_response(prompt):
    try:
        # Call the predict method with prompt and other parameters
        result = client.predict(
            prompt=prompt,
            temperature=0.7,
            max_new_tokens=300,
            top_p=0.9,
            repetition_penalty=1.2,
            api_name="/chat"  # Specify the API name for the chat endpoint
        )
        response = result.strip()  # Adjust this according to the actual response structure
        return response
    except Exception as e:
        return f"Error getting response: {e}"

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    if 'file' not in request.files or 'company_url' not in request.form:
        return jsonify({'error': 'No file uploaded or no company URL provided'}), 400

    file = request.files['file']
    company_url = request.form['company_url']  # Update variable name to `company_url`
    resume_text = extract_text_from_pdf(file)
    company_details = fetch_company_details_from_url(company_url)

    prompt = (
        "You are an HR expert tasked with creating interview questions tailored to a specific job role. "
        "Using the following information, generate a set of insightful and targeted interview questions. "
        "The questions should be designed to assess the candidate's suitability for the role based on their experience, "
        "skills, and qualifications as listed in the resume, and should also consider the specifics of the company, including "
        "its industry, values, and any relevant details you can infer from the provided company information.\n\n"
        "Resume:\n{resume_text}\n\nCompany Details:\n{company_details}\n\nQuestions:"
    ).format(resume_text=resume_text, company_details=company_details)

    questions = get_response(prompt)
    question_list = [q.strip() for q in questions.split('\n') if q.strip()]

    return jsonify({'questions': question_list, 'resume_text': resume_text, 'company_details': company_details})

@app.route('/generate_follow_up', methods=['POST'])
def generate_follow_up():
    data = request.json

    if 'question' not in data or 'response' not in data or 'resume_text' not in data:
        return jsonify({'error': 'Missing "question", "response" or "resume_text" key in request data'}), 400

    question = data['question']
    response = data['response']
    resume_text = data['resume_text']

    # Adjust the prompt based on the response length
    if len(response.split()) < 5:  # Assuming a short response is less than 5 words
        prompt = (
            "The following candidate provided a very brief response to a question. "
            "Please give feedback on the importance of providing detailed answers in an interview \n\n"
            "Question: {question}\nCandidate's Response: {response}\nResume:\n{resume_text}\n\n"

        ).format(question=question, response=response, resume_text=resume_text)
    else:
        prompt = (
            "Based on the following response from the candidate to a specific question, generate one "
            "creative follow-up questions to based on the response be more like a hr itself.\n\n"
            "Question: {question}\nCandidate's Response: {response}\nResume:\n{resume_text}\n\nFollow-Up Questions:"
        ).format(question=question, response=response, resume_text=resume_text)

    response = get_response(prompt)
    follow_up_questions = [q.strip() for q in response.split('\n') if q.strip()]
    return jsonify({'follow_up_questions': follow_up_questions})

@app.route('/generate_feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    interview_history = data['interview_history']

    prompt = (
        "Based on the following interview history, provide feedback on the candidate's performance. "
        "Highlight their strengths and areas for improvement to help them perform better in real-life interviews.\n\n"
        "Interview History:\n{interview_history}\n\nFeedback:"
    ).format(interview_history=interview_history)
    feedback = get_response(prompt)

    return jsonify({'feedback': feedback})

@app.route('/tts', methods=['POST'])
def generate_tts():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    return send_file("output.mp3", mimetype='audio/mp3')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
    return resume_text

def fetch_company_details_from_url(company_url):
    try:
        response = requests.get(company_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant details from the company webpage
        # This example extracts meta description and title
        title = soup.find('title').get_text() if soup.find('title') else 'No title found'
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description['content'] if meta_description else 'No meta description found'

        return f"Title: {title}\nDescription: {description}"
    except requests.RequestException as e:
        return f"Error fetching company details: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
