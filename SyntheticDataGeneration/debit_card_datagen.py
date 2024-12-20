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
    "You are an expert in generating realistic and detailed debit card applications "
    "for synthetic data purposes. Each application must include complete details about "
    "checking accounts, reflect professional language, and vary in structure to cover "
    "different banks. Just generate these examples, don't provide any extra information."
)

debit_card_template = """
**Application Type**: Debit Card
**Bank Name**: {bank_name}
**Applicant Name**: {name}
**Address**: {address}
**Phone Number**: {phone}
**Email Address**: {email}
**Employment Status**: {employment}
**Checking Account Type**: {account_type}
**Initial Deposit**: {initial_deposit}
**Preferred ATM Network**: {atm_network}
**Additional Services**: {additional_services}
**ID Type**: {id_type}
**ID Number**: {id_number}
"""

user_prompt = (
    f"Generate 5 distinct and detailed debit card applications using the following template:\n\n"
    f"{debit_card_template}\n\n"
    "1. Use realistic applicant details such as names, addresses, phone numbers, email addresses, "
    "employment information, and checking account details.\n"
    "2. Ensure that the applications are diverse, considering different banks, account types "
    "(basic checking, premium checking, student checking), and regions.\n"
    "3. Include realistic initial deposits and ATM network preferences."
)

def generate_applications():
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)
    applications = response.content.strip().split("\n\n---\n\n")
    return applications

def save_to_csv(applications, filename="generated_debit_card_applications.txt"):
    df = pd.DataFrame({"Application": applications})
    df.to_csv(filename, index=False)
    print(f"Generated applications saved to {filename}")

if __name__ == "__main__":
    applications = generate_applications()
    save_to_csv(applications) 