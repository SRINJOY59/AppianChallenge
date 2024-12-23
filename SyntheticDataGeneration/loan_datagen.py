import json
import random
import csv
from faker import Faker
# List of additional Loan Templates


from Templates_bank.loan_templates import (
    sbi_loan_template,
    hdfc_loan_template,
    icici_loan_template,
    axis_loan_template,
    kotak_loan_template,
    pnb_loan_template,
    yes_bank_loan_template,
    canara_loan_template,
    union_bank_loan_template,
    bob_loan_template,
)


# Initialize Faker
fake = Faker()

# List of Loan Templates
loan_templates = [
    sbi_loan_template,
    hdfc_loan_template,
    icici_loan_template,
    axis_loan_template,
    kotak_loan_template,
    pnb_loan_template,
    yes_bank_loan_template,
    canara_loan_template,
    union_bank_loan_template,
    bob_loan_template
]

# Generate synthetic loan data from template
def generate_data_from_template(template):
    synthetic_data = {
        # Basic Personal Information
        "ref_number": fake.uuid4(),
        "branch_code": fake.random_number(digits=5),
        "name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d"),
        "pan": fake.random_uppercase_letter() + str(fake.random_number(digits=4)),
        "aadhaar": str(fake.random_number(digits=12)),
        "mobile": fake.phone_number(),
        "email": fake.email(),
        
        # Residence Details
        "current_address": fake.address(),
        "city": fake.city(),
        "pin": fake.zipcode(),
        "residence_status": random.choice(["Owned", "Rented", "Company Provided"]),
        "years_at_residence": random.randint(1, 20),
        
        # Employment Details
        "occupation": random.choice(["Salaried", "Self-Employed", "Business"]),
        "company_name": fake.company(),
        "annual_income": random.choice([300000, 500000, 800000, 1000000, 1500000]),
        "work_experience": random.randint(1, 30),
        
        # Loan Details
        "loan_type": random.choice(["Home", "Personal", "Vehicle", "Education"]),
        "loan_amount": random.choice([50000, 100000, 200000, 500000, 1000000]),
        "loan_tenure": random.choice([5, 10, 15, 20]),
        "loan_purpose": random.choice(["Home Purchase", "Car Purchase", "Debt Consolidation", "Education"]),
        
        # Existing Loans
        "has_existing_loans": random.choice(["Yes", "No"]),
        "current_emi": random.choice([1000, 5000, 10000, 15000, 20000]),
        
        # Financial Details for templates
        "monthly_income": random.choice([25000, 35000, 45000, 60000]),
        "total_experience": random.randint(1, 20),
        "office_address": fake.address(),
        "required_amount": random.choice([100000, 300000, 500000]),
        "tenure_months": random.choice([12, 24, 36, 60]),
        "other_income": random.choice(["Investments", "Rent", "Freelancing"]),
        "account_number": fake.random_number(digits=10),
        "existing_emis": random.choice([0, 1, 2]),
        
        # Other Details for ICICI template
        "primary_bank": random.choice(["SBI", "HDFC", "ICICI", "Axis", "Kotak"]),
        "account_type": random.choice(["Savings", "Current", "Fixed Deposit"]),
        "loan_category": random.choice(["Personal", "Home", "Auto", "Business"]),
        "requested_amount": random.choice([100000, 300000, 500000]),
        "loan_purpose": random.choice(["Home Purchase", "Business Expansion", "Car Purchase"]),
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

# Generate data points for each loan template
all_data = []
for template in loan_templates:
    for _ in range(2):  # Loop to generate two data points for each template
        data_points = generate_data_from_template(template)
        if data_points:
            all_data.append({
                "information": data_points,  # Using the formatted template as string
                "label": "loan",
                "specific_label": "loan details"
            })

# Write to CSV file
output_file = r"Data/loan_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["information", "label", "specific_label"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Synthetic loan data saved to {output_file}")
