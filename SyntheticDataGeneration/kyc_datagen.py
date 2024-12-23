import json
import random
import csv
from faker import Faker
from Templates_bank.kyc_templates import (
    sbi_kyc_template,
    hdfc_kyc_template,
    icici_kyc_template,
    axis_kyc_template,
    kotak_kyc_template,
    pnb_kyc_template,
    yes_bank_kyc_template,
    canara_kyc_template,
    union_bank_kyc_template,
    bob_kyc_template,
)

fake = Faker()

# List of KYC templates
kyc_templates = [
    sbi_kyc_template,
    hdfc_kyc_template,
    icici_kyc_template,
    axis_kyc_template,
    kotak_kyc_template,
    pnb_kyc_template,
    yes_bank_kyc_template,
    canara_kyc_template,
    union_bank_kyc_template,
    bob_kyc_template,
]

# Generate synthetic data from template
def generate_data_from_template(template):
    synthetic_data = {
        # Basic Personal Information
        "ref_number": fake.uuid4(),
        "form_number": fake.random_number(digits=6),
        "customer_name": fake.name(),
        "guardian_name": fake.name(),
        "mother_name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d"),
        "gender": random.choice(["Male", "Female", "Other"]),
        "marital_status": random.choice(["Single", "Married", "Divorced"]),
        "nationality": fake.country(),
        
        # Identity Documents
        "pan": fake.random_uppercase_letter() + str(fake.random_number(digits=4)),
        "aadhaar": str(fake.random_number(digits=12)),
        "passport_no": fake.random_uppercase_letter() + str(fake.random_number(digits=6)),
        "driving_license": fake.random_uppercase_letter() + str(fake.random_number(digits=10)),
        "voter_id": fake.random_uppercase_letter() + str(fake.random_number(digits=10)),
        
        # Contact Details
        "mobile": fake.phone_number(),
        "landline": fake.phone_number(),
        "email": fake.email(),
        
        # Address Details
        "current_address": fake.address(),
        "permanent_address": fake.address(),
        "city": fake.city(),
        "state": fake.state(),
        "pin": fake.zipcode(),
        "residence_type": random.choice(["Owned", "Rented", "Leased"]),
        
        # Occupation Details
        "occupation": random.choice(["Employed", "Self-Employed", "Student", "Retired"]),
        "income_range": random.choice(["Below 2 Lakh", "2-5 Lakh", "5-10 Lakh", "Above 10 Lakh"]),
        "income_source": random.choice(["Salary", "Business", "Investments", "Other"]),
        "employer": fake.company(),
        
        # Account Details
        "account_type": random.choice(["Savings", "Current", "Fixed Deposit"]),
        "account_number": fake.random_number(digits=10),
        "branch_code": fake.random_number(digits=5),
    }

    # Find all placeholders in the template (anything within curly braces)
    placeholders = [key.strip('{}') for key in template.split() if key.startswith("{") and key.endswith("}")]

    # Create a dictionary to handle missing attributes
    data_to_format = {key: synthetic_data.get(key, '') for key in placeholders}

    try:
        filled_template = template.format(**data_to_format)
    except KeyError as e:
        print(f"Missing attribute: {e}, leaving it vacant.")
        filled_template = template.format(**{key: synthetic_data.get(key, '') for key in placeholders})

    return filled_template

all_data = []
for template in kyc_templates:
    for _ in range(2):  # Loop to generate two data points for each template
        data_points = generate_data_from_template(template)
        if data_points:
            all_data.append({
                "information": data_points,  # Using the formatted template as string
                "label": "bank",
                "specific_label": "kyc details"
            })

# Write to CSV file
output_file = r"Data/kyc_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["information", "label", "specific_label"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Synthetic data saved to {output_file}")
