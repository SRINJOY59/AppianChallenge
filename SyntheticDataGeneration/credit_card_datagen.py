import json
import random
import csv
from faker import Faker
from Templates_bank.credit_card_template import (
    sbi_credit_card_template,
    hdfc_credit_card_template,
    icici_credit_card_template,
    axis_credit_card_template,
    kotak_credit_card_template,
    pnb_credit_card_template,
    yes_bank_credit_card_template,
    canara_credit_card_template,
    union_bank_credit_card_template,
    bob_credit_card_template,
)

# Initialize Faker
fake = Faker()

# List of templates
templates = [
    sbi_credit_card_template,
    hdfc_credit_card_template,
    icici_credit_card_template,
    axis_credit_card_template,
    kotak_credit_card_template,
    pnb_credit_card_template,
    yes_bank_credit_card_template,
    canara_credit_card_template,
    union_bank_credit_card_template,
    bob_credit_card_template,
]

# Generate synthetic data from template
def generate_data_from_template(template):
    synthetic_data = {
        # Basic Personal Information
        "ref_number": fake.uuid4(),
        "branch_code": fake.random_number(digits=6),
        "name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d"),
        "pan": fake.random_uppercase_letter() + str(fake.random_number(digits=4)),  # Fix applied here
        "aadhaar": fake.random_number(digits=12),
        "mobile": fake.phone_number(),
        "email": fake.email(),

        # Residence Information
        "current_address": fake.address(),
        "permanent_address": fake.address(),
        "city": fake.city(),
        "state": fake.state(),
        "country": fake.country(),
        "pin": fake.zipcode(),
        "residence_status": random.choice(["Owned", "Rented", "Company Provided"]), 

        # Employment Information
        "occupation": random.choice(["Salaried", "Self-Employed", "Business"]),
        "company_name": fake.company(),
        "annual_income": fake.random_int(min=300000, max=2000000),
        "work_experience": random.randint(1, 20),

        # Financial Details
        "credit_score": random.randint(300, 850),
        "previous_credit_card": random.choice([True, False]),
        "monthly_expenses": fake.random_int(min=5000, max=50000),
        "loan_type": random.choice(["Home Loan", "Car Loan", "Personal Loan", "None"]),
        "savings_balance": fake.random_int(min=10000, max=500000),
        "loan_balance": fake.random_int(min=5000, max=100000),
        "existing_credit_limit": fake.random_int(min=50000, max=500000),

        # Bank Details
        "account_number": fake.bban(),
        "account_type": random.choice(["Savings", "Current"]),
        "branch_name": fake.city(),
        "account_opening_date": fake.date_this_decade().strftime("%Y-%m-%d"),

        # Card Preferences
        "card_type": random.choice(["Simply Save", "Elite", "Prime", "Signature"]),
        "secondary_email": fake.email(),
        "referral_code": fake.uuid4(),
    }

    # Find all placeholders in the template (anything within curly braces)
    placeholders = [key.strip('{}') for key in template.split() if key.startswith("{") and key.endswith("}")]

    # Create a dictionary to handle missing attributes
    data_to_format = {key: synthetic_data.get(key, '') for key in placeholders}

    try:
        filled_template = template.format(**data_to_format)
    except KeyError as e:
        print(f"Missing attribute: {e}, leaving it vacant.")
        # Replace missing keys with empty string
        filled_template = template.format(**{key: synthetic_data.get(key, '') for key in placeholders})

    return filled_template




# Generate two data points for each template
all_data = []
for template in templates:
    for _ in range(2):  # Loop to generate two data points for each template
        data_points = generate_data_from_template(template)
        if data_points:
            all_data.append({
                "information": data_points,  # Using the formatted template as string
                "label": "bank",
                "specific_label": "credit card details"
            })

# Write to CSV file
output_file = r"Data/credit_card_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["information", "label", "specific_label"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Synthetic data saved to {output_file}")
