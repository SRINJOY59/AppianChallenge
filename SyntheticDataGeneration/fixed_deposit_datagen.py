import json
import random
import csv
from faker import Faker
from Templates_bank.fixed_deposit_templates import (
    sbi_fd_template,
    hdfc_fd_template,
    icici_fd_template,
    axis_fd_template,
    kotak_fd_template,
    pnb_fd_template,
    yes_bank_fd_template,
    canara_fd_template,
    union_bank_fd_template,
    bob_fd_template,
)

# Initialize Faker
fake = Faker()

# List of Fixed Deposit templates
fd_templates = [
    sbi_fd_template,
    hdfc_fd_template,
    icici_fd_template,
    axis_fd_template,
    kotak_fd_template,
    pnb_fd_template,
    yes_bank_fd_template,
    canara_fd_template,
    union_bank_fd_template,
    bob_fd_template,
]

# Generate synthetic data from template
def generate_data_from_template(template):
    synthetic_data = {
        # Basic Personal Information
        "ref_number": fake.uuid4(),
        "form_number": fake.random_number(digits=6),
        "name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d"),
        "pan": fake.random_uppercase_letter() + str(fake.random_number(digits=4)),
        "mobile": fake.phone_number(),
        "email": fake.email(),

        # Deposit Details
        "deposit_amount": fake.random_int(min=10000, max=500000),
        "deposit_period": f"{random.randint(1, 10)} Years",
        "interest_rate": random.uniform(5, 8),  # in percentage
        "deposit_type": random.choice(["Regular", "Tax Saving", "Senior Citizen"]),
        "interest_payout": random.choice(["Monthly", "Quarterly", "Annual", "Cumulative"]),

        # Account Details
        "source_account": fake.bban(),
        "branch_name": fake.city(),
        "operation_mode": random.choice(["Single", "Joint", "Either or Survivor"]),

        # Maturity Instructions
        "maturity_instruction": random.choice(["Auto Renew", "Credit to Account"]),
        "renewal_period": random.choice(["1 Year", "2 Years", "3 Years"]),
        "credit_account": fake.bban(),

        # Nominee Details
        "nominee_name": fake.name(),
        "nominee_relation": random.choice(["Spouse", "Child", "Parent", "Sibling"]),
        "nominee_dob": fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%Y-%m-%d"),
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

# Generate two data points for each Fixed Deposit template
all_data = []
for template in fd_templates:
    for _ in range(2):  # Loop to generate two data points for each template
        data_points = generate_data_from_template(template)
        if data_points:
            all_data.append({
                "information": data_points,  # Using the formatted template as string
                "label": "bank",
                "specific_label": "fixed deposit details"
            })

# Write to CSV file
output_file = r"Data/fixed_deposit_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["information", "label", "specific_label"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Synthetic data saved to {output_file}")
