import streamlit as st
import json
import time
import os
import tempfile
from collections import Counter
from KGAgent.KnowledgeGraphAgent import KnowledgeGraphAgent, load_as_json
from Specific_Agents.bank_specification_agent import DocumentSummaryAgentBank
from Specific_Agents.financial_specification_agent import FinancialSummaryAgent
from Specific_Agents.identity_specification_agent import IdentityClassificationAgent
from Specific_Agents.receipt_specification_agent import ReceiptClassificationAgent
from Main_Clf_Agents.base_ml_model import load_models_and_predict
from Main_Clf_Agents.mistral_base_agent import MistralBaseAgent
from Text_Extraction.parser_llama import extract_text_from_llama_parse
from Text_Extraction.parser_pdfminer import extract_text_from_pdf
from Text_Extraction.parser_groundX import initialize_groundx_client, parse_with_groundx
from Text_Extraction.scan_checker import is_scanned_pdf
from Main_Clf_Agents.gemini_base_agent import DocumentCategoryAgentGemini
from Email_services.email_service import EmailService


def save_user_data(username, email):
    try:
        with open('USER_DATA/user_data.json', 'r') as file:
            user_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        user_data = []
        
    email_sender = EmailService()
    html_template = """
    <!DOCTYPE html>
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                
                <h2 style="color: #2c3e50; margin-bottom: 20px;">Welcome to FinDoc.ai! 🎉</h2>
                
                <p style="color: #34495e;">Dear {name},</p>
                
                <p style="color: #34495e;">Thank you for choosing FinDoc.ai! We're excited to have you on board. Your account has been successfully created and you're all set to start using our intelligent document processing services.</p>
                
                <div style="background-color: #ffffff; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #2c3e50; margin-top: 0;">Quick Start Guide:</h3>
                    <ul style="color: #34495e;">
                        <li>Upload your documents securely</li>
                        <li>Choose between Quick or Premium processing</li>
                        <li>Get instant analysis and insights</li>
                        <li>Access your documents anywhere, anytime</li>
                    </ul>
                </div>

                <p style="color: #34495e;">We value your feedback! Help us improve by sharing your experience:</p>
                
                <div style="text-align: center; margin: 25px 0;">
                    <a href="https://docs.google.com/forms/d/e/1FAIpQLSfA_v4mWcH4LI2jb-axOXexXqvL7tUbAH9xcnDCpD2NMOyAww/viewform?usp=dialog" 
                       style="background-color: #3498db; 
                              color: white; 
                              padding: 12px 25px; 
                              text-decoration: none; 
                              border-radius: 5px;
                              font-weight: bold;">
                        Share Your Feedback
                    </a>
                </div>

                <p style="color: #34495e;">Need help? Our support team is always here to assist you. Simply reply to this email!</p>

                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
                    <p style="color: #7f8c8d; margin: 0;">Best regards,</p>
                    <p style="color: #2c3e50; font-weight: bold; margin: 5px 0;">The FinDoc.ai Team</p>
                </div>
                
                <div style="margin-top: 20px; font-size: 12px; color: #7f8c8d;">
                    <p>Follow us on social media:</p>
                    <a href="#" style="color: #3498db; text-decoration: none; margin-right: 10px;">Twitter</a>
                    <a href="#" style="color: #3498db; text-decoration: none; margin-right: 10px;">LinkedIn</a>
                    <a href="#" style="color: #3498db; text-decoration: none;">Facebook</a>
                </div>
            </div>
        </body>
    </html>
    """
    if not any(user['username'] == username for user in user_data):
        user_data.append({"username": username, "email": email})
        email_sender.send_email(recipient_email=email, subject="Welcome to FinDoc.ai", html_content=html_template.format(name=username))

    with open('USER_DATA/user_data.json', 'w') as file:
        json.dump(user_data, file, indent=4)

def save_user_history(query, relevant_info):
    try:
        with open('USER_DATA/user_history.json', 'r') as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    username = st.session_state.username
    email = st.session_state.email
    
    history.append({
        "username": username,
        "email": email,
        "query": query,
        "relevant_information": relevant_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    with open('USER_DATA/user_history.json', 'w') as file:
        json.dump(history, file, indent=4)

def display_user_history():
    try:
        with open('USER_DATA/user_history.json', 'r') as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    current_user = st.session_state.username
    user_history = [entry for entry in history if entry.get('username') == current_user]

    st.sidebar.markdown("### Your Chat History")
    
    if not user_history:
        st.sidebar.write("No history found")
    else:
        for entry in user_history:
            info_text = ", ".join(f"{key}: {value}" for key, value in entry['relevant_information'].items())
            st.sidebar.markdown(info_text[:50])
            st.sidebar.write("---")

def show_progress(message, sleep_time=2):
    placeholder = st.empty()
    with placeholder.container():
        st.write(message)
    time.sleep(sleep_time)
    placeholder.empty()

def add_blue_theme():
    st.markdown(
        """
        <style>
        body {
            background-color: #e0f5ff;  /* Light Blue Background */
            color: #001f3f;  /* Dark Blue Text */
        }
        .stButton>button {
            background-color: #002952;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 18px;
        }
        .stButton>button:hover {
            background-color: #003366;
        }
        .stTextInput>div>input {
            background-color: #ffffff;
            color: #001f3f;
            border: 2px solid #002952;
            border-radius: 6px;
        }
        .stSpinner {
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #001f3f;  /* Dark Blue for Titles */
        }
        .css-1d391kg {
            background-color: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="FinDoc.ai", layout="wide", page_icon="🔖")

def main():
    if "username" not in st.session_state or "email" not in st.session_state:
        st.title("🔖 FinDoc.ai: Intelligent Document Parsing and Knowledge Extraction")
        st.markdown("### Please enter your details to get started.")

        with st.container():

            with st.form(key="user_form"):
                username = st.text_input("Enter your Username")
                email = st.text_input("Enter your Email Address")
                submit_button = st.form_submit_button("Submit")

                if submit_button:
                    if username and email:
                        st.session_state.username = username
                        st.session_state.email = email
                        save_user_data(username, email)
                        st.session_state.form_submitted = True
                    else:
                        st.warning("Please enter both username and email.")
    else:
        if "form_submitted" not in st.session_state or not st.session_state.form_submitted:
            return  
        
        add_blue_theme()
        tmp_path = r'Test_PDFs\voter_card.pdf'
        st.markdown("### Welcome to the FinDoc.ai!")
        st.markdown("### Effortlessly upload, analyze, and extract insights from your documents with precision.")
        uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
        st.info("""
            ### Choose Your Processing Mode 🚀

            **1. GroundX Mode**
            - ✨ Highest accuracy and detail
            - ⏱️ Takes longer to process

            **2. Llama Mode**
            - ⚡ Fast processing speed
            - 📊 Good accuracy for most documents
 

            _Select the mode that best suits your needs!
        """)
        mode = st.selectbox("Select Processing Mode", options=["groundx", "llama"])
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
        
        is_scanned = is_scanned_pdf(tmp_path)
        if is_scanned:
            if mode == "llama":
                sample_text = extract_text_from_llama_parse(tmp_path)
            else:
                bucket_name = "sample_bucket"
                api_key = os.getenv("GROUNDX_API_KEY")
                groundx_client = initialize_groundx_client(api_key)
                file_name = tmp_path.split("\\")[-1]
                file_path = tmp_path
                file_type = "pdf"
                sample_text = parse_with_groundx(groundx_client, bucket_name, file_name, file_path, file_type)
        else:
            sample_text = extract_text_from_pdf(tmp_path)

        
        print(sample_text)

        if st.button("Process Document"):
            if not sample_text.strip():
                st.warning("Please enter the document text to proceed.")
            else:
                with st.spinner("Classifying document... This may take a moment."):
                    
                    show_progress("Step 1: Predicting the document type using the base models...")

                    prediction = load_models_and_predict(sample_text) 
                    
                    show_progress("Step 2: Predicting the document type using Mixtral...")

                    mistral_agent = MistralBaseAgent()
                    mistral_prediction = mistral_agent.classify_document(sample_text) 
                    
                    show_progress("Step 3: Predicting the document type using Gemini...")

                    gemini_agent = DocumentCategoryAgentGemini()
                    gemini_prediction = gemini_agent.categorize_document(sample_text)  

                    predictions = [prediction, mistral_prediction, gemini_prediction]
                    most_common_prediction = Counter(predictions).most_common(1)
                    base_prediction = most_common_prediction[0][0] if most_common_prediction else "Uncategorized"

                    st.success(f"Document classified as: **{base_prediction}**")

                with st.spinner("Generating knowledge graph..."):
                    show_progress("Step 4: Creating knowledge graph based on classification...")
                    kg_agent = KnowledgeGraphAgent()
    
                    knowledge_graph = kg_agent.generate_knowledge_graph(sample_text, graph_type=base_prediction)  # Replace with actual function
                    knowledge_graph_json = load_as_json(knowledge_graph)

                    st.success("Knowledge graph created successfully!")

                    save_user_history(sample_text, knowledge_graph_json.get('relevant_information', 'No relevant information'))
                    
                    if base_prediction == "bank":
                        bank_agent = DocumentSummaryAgentBank()
                        bank_summary = bank_agent.generate_summary(knowledge_graph)
                        st.markdown(bank_summary)
                    elif base_prediction == "financial":
                        financial_agent = FinancialSummaryAgent()
                        financial_summary = financial_agent.generate_summary(knowledge_graph)
                        st.markdown(financial_summary)
                    elif base_prediction == "identity":
                        identity_agent = IdentityClassificationAgent()
                        identity_summary = identity_agent.classify_identity(knowledge_graph)
                        st.markdown(identity_summary)   
                    elif base_prediction == "receipt":
                        receipt_agent = ReceiptClassificationAgent()
                        receipt_summary = receipt_agent.classify_receipt(knowledge_graph)
                        st.markdown(receipt_summary)
                    else:
                        st.markdown("No relevant information found")

                display_user_history()

if __name__ == "__main__":
    main()
