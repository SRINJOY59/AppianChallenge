import json
import random
import csv
from faker import Faker
from Templates_bank.passbook_template import (
    sbi_passbook_template,
    hdfc_passbook_template,
    icici_passbook_template,
    axis_passbook_template,
    kotak_passbook_template,
    pnb_passbook_template,
    yes_bank_passbook_template, 
    canara_passbook_template, 
    union_bank_passbook_template, 
    bob_passbook_template
)

fake = Faker()

passbook_templates = [
    sbi_passbook_template,
    hdfc_passbook_template,
    icici_passbook_template,
    axis_passbook_template,
    kotak_passbook_template,
    pnb_passbook_template,
    yes_bank_passbook_template, 
    canara_passbook_template, 
    union_bank_passbook_template, 
    bob_passbook_template
]

# Generate synthetic passbook data from template
def generate_passbook_data_from_template(template):
    synthetic_data = {
        # Basic Account Information
        "branch_name": fake.company(),
        "ifsc_code": fake.bban(),
        
        # Account Details
        "account_holder": fake.name(),
        "account_number": str(fake.random_number(digits=10)),
        "account_type": random.choice(["Savings", "Current", "NRI"]),
        "customer_id": str(fake.random_number(digits=10)),
        "opening_date": fake.date_this_century().strftime("%Y-%m-%d"),
        
        # Transaction Details
        "date": fake.date_this_year().strftime("%Y-%m-%d"),
        "value_date": fake.date_this_year().strftime("%Y-%m-%d"),
        "description": fake.word(),
        "cheque_no": fake.random_number(digits=6),
        "debit": random.choice([1000, 5000, 10000, 15000, 20000]),
        "credit": random.choice([500, 2000, 10000, 15000]),
        "balance": random.choice([10000, 20000, 50000, 100000]),
        
        # HDFC-specific attributes
        "operation_mode": random.choice(["Single", "Joint"]),
        "mobile_number": fake.phone_number(),
        
        # ICICI-specific attributes
        "customer_name": fake.name(),
        "ckyc_number": fake.uuid4(),
        
        # Kotak-specific attributes
        "home_branch": fake.company(),
        "scheme_code": fake.random_number(digits=5),
        "crn": fake.random_number(digits=8),
        
        "branch_code": fake.random_number(digits=5),
        "instrument_no": fake.random_number(digits=10),
        
        "particulars": fake.word(),
        "ref_no": fake.random_number(digits=6),
        "withdrawal": random.choice([0, 1000, 5000, 10000]),
        "deposit": random.choice([500, 2000, 5000]),
        "balance": random.choice([10000, 20000, 50000]),
    }

    placeholders = [key.strip('{}') for key in template.split() if key.startswith("{") and key.endswith("}")]

    data_to_format = {key: synthetic_data.get(key, '') for key in placeholders}

    try:
        filled_template = template.format(**data_to_format)
    except KeyError as e:
        print(f"Missing attribute: {e}, leaving it vacant.")
        filled_template = template.format(**{key: synthetic_data.get(key, '') for key in placeholders})

    return filled_template

all_data = []
for template in passbook_templates:
    for _ in range(2):  
        data_points = generate_passbook_data_from_template(template)
        if data_points:
            all_data.append({
                "information": data_points,  
                "label": "passbook",
                "specific_label": "passbook details"
            })

output_file = r"Data/passbook_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["information", "label", "specific_label"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Synthetic passbook data saved to {output_file}")
