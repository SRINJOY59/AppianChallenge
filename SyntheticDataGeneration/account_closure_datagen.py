import random
import csv
from faker import Faker
from Templates_bank.account_closure_templates import (
    sbi_closure_template,
    hdfc_closure_template,
    icici_closure_template,
    axis_closure_template,
    kotak_closure_template,
    pnb_closure_template,
    yes_bank_closure_template,
    canara_closure_template,
    union_bank_closure_template,
    bob_closure_template,
)

# Initialize Faker
fake = Faker()

# List of templates for account closure
account_closure_templates = [
    sbi_closure_template,
    hdfc_closure_template,
    icici_closure_template,
    axis_closure_template,
    kotak_closure_template,
    pnb_closure_template,
    yes_bank_closure_template,
    canara_closure_template,
    union_bank_closure_template,
    bob_closure_template,
]

# Generate synthetic data from account closure template
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

        # Address Information
        "current_address": fake.address(),
        "permanent_address": fake.address(),
        "city": fake.city(),
        "state": fake.state(),
        "country": fake.country(),
        "pin": fake.zipcode(),

        # Account Information
        "account_number": fake.bban(),
        "account_type": random.choice(["Savings", "Current"]),
        "branch_name": fake.city(),
        "account_opening_date": fake.date_this_decade().strftime("%Y-%m-%d"),

        # Closure Details
        "closure_reason": random.choice(["Personal Reasons", "Account No Longer Needed", "Better Services Elsewhere"]),
        "settlement_mode": random.choice(["Transfer", "Cheque", "Cash"]),
        "transfer_bank_name": fake.company(),
        "transfer_account_number": fake.bban(),
        "transfer_ifsc": fake.random_number(digits=11),
        "debit_card": fake.random_number(digits=16),
        "internet_banking": random.choice(["Active", "Inactive"]),
        "mobile_banking": random.choice(["Active", "Inactive"]),
        "standing_instructions": random.choice(["Yes", "No"]),
        "outstanding_charges": random.choice(["Yes", "No"]),
        "unused_cheques": random.randint(1, 5),
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

# Generate two data points for each account closure template
all_data = []
for template in account_closure_templates:
    for _ in range(2):  # Loop to generate two data points for each template
        data_points = generate_data_from_template(template)
        if data_points:
            all_data.append({
                "information": data_points,  # Using the formatted template as string
                "label": "bank",
                "specific_label": "account closure details"
            })

# Write to CSV file
output_file = r"Data/account_closure_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["information", "label", "specific_label"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Synthetic data saved to {output_file}")
