from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=1500,
    timeout=60,
    max_retries=2,
)

system_prompt = (
    "You are an expert in generating realistic and detailed credit card applications "
    "for synthetic data purposes. Each application must include complete details, "
    "reflect professional language, and vary in structure to cover different banks."
    "Just generate these examples, don't provide any extra information."
)

credit_card_template = """
**Application Type**: Credit Card
**Bank Name**: Global Bank
**Applicant Name**: {name}
**Address**: {address}
**Phone Number**: {phone}
**Email Address**: {email}
**Monthly Income**: {income}
**Credit Card Type**: {card_type}
**Credit Limit Requested**: {credit_limit}
**Additional Notes**: {additional_notes}
"""

user_prompt = (
    f"Generate 5 distinct and detailed credit card applications using the following template:\n\n"
    f"{credit_card_template}\n\n"
    "1. Use realistic applicant details such as names, addresses, phone numbers, email addresses, financial information, "
    "and credit card details.\n"
    "2. Ensure that the applications are diverse, considering different banks, applicant profiles, and regions."
)

def generate_applications():
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)

    applications = response.content.strip().split("\n\n---\n\n")  
    return applications

def save_to_csv(applications, filename="generated_credit_card_applications.txt"):
    df = pd.DataFrame({"Application": applications})
    df.to_csv(filename, index=False)
    print(f"Generated applications saved to {filename}")

if __name__ == "__main__":
    applications = generate_applications()
    save_to_csv(applications)
