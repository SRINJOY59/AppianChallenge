# SBI Credit Card Application Template
sbi_credit_card_template = """
**STATE BANK OF INDIA - CREDIT CARD APPLICATION**
Application Reference: {ref_number}
Branch Code: {branch_code}

PERSONAL DETAILS
-----------------
Applicant Name: {name}
Date of Birth: {dob}
PAN Card Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {mobile}
Email ID: {email}

RESIDENCE DETAILS
----------------
Current Address: {current_address}
City: {city}
PIN Code: {pin}
Residence Status: {residence_status}  # Owned/Rented/Company Provided

EMPLOYMENT DETAILS
-----------------
Occupation Type: {occupation}  # Salaried/Self-Employed/Business
Company Name: {company_name}
Annual Income: {annual_income}
Years in Current Job: {work_experience}

BANK DETAILS
-----------
SBI Account Number: {account_number}
Account Type: {account_type}
Branch Name: {branch_name}
Account Opening Date: {account_opening_date}

CARD PREFERENCES
---------------
Card Type Requested: {card_type}  # Simply Save/Elite/Prime/Signature
"""

# HDFC Credit Card Application Template
hdfc_credit_card_template = """
**HDFC BANK - CREDIT CARD APPLICATION**
Application ID: {application_id}

APPLICANT INFORMATION
--------------------
Full Name: {full_name}
Date of Birth: {dob}
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Contact Details:
    Mobile: {mobile}
    Email: {email}
    
RESIDENTIAL INFORMATION
----------------------
Address: {address}
City & PIN: {city_pin}
Years at Current Address: {years_at_address}

PROFESSIONAL DETAILS
-------------------
Employment Type: {employment_type}
Employer Name: {employer}
Monthly Income: {monthly_income}
Office Address: {office_address}

BANKING RELATIONSHIP
-------------------
Existing HDFC Account: {has_hdfc_account}  # Yes/No
Account Number: {account_number}
Relationship Duration: {relationship_years}

CARD SELECTION
-------------
Preferred Card Variant: {card_variant}  # Moneyback/Regalia/Diners Club
Reward Program Choice: {reward_choice}
"""

# ICICI Credit Card Application Template
icici_credit_card_template = """
**ICICI BANK - CREDIT CARD APPLICATION**
Reference Number: {ref_no}

PERSONAL INFORMATION
-------------------
Customer Name: {customer_name}
Father's Name: {father_name}
Mother's Name: {mother_name}
Date of Birth: {dob}
Gender: {gender}
Marital Status: {marital_status}

IDENTIFICATION
-------------
PAN: {pan_number}
Aadhaar: {aadhaar_number}
Voter ID/Driving License: {id_number}

CONTACT DETAILS
--------------
Mobile Number: {mobile}
Alternate Number: {alternate_mobile}
Email Address: {email}
Residence Address: {residence_address}
PIN Code: {pin_code}

EMPLOYMENT & INCOME
------------------
Occupation: {occupation}
Company Name: {company}
Designation: {designation}
Work Experience: {experience}
Gross Annual Income: {annual_income}
Net Monthly Income: {monthly_income}

EXISTING RELATIONSHIP
--------------------
ICICI Account Holder: {is_account_holder}  # Yes/No
Account Number: {account_number}
Credit Bureau Score: {cibil_score}

CARD DETAILS
-----------
Card Variant Applied: {card_variant}  # Coral/Rubyx/Sapphiro/Diamond
Add-On Card Required: {addon_card}  # Yes/No
"""

# Axis Bank Credit Card Application Template
axis_credit_card_template = """
**AXIS BANK - CREDIT CARD APPLICATION**
Application Number: {app_number}

PERSONAL DETAILS
---------------
Applicant's Name: {applicant_name}
Date of Birth: {dob}
PAN Number: {pan}
Aadhaar Number: {aadhaar}

CONTACT INFORMATION
------------------
Mobile: {mobile_number}
Email: {email_id}
Current Address: {current_address}
Permanent Address: {permanent_address}

PROFESSIONAL DETAILS
-------------------
Employment Status: {employment_status}
Current Employer: {employer_name}
Monthly Salary: {monthly_salary}
Office Address: {office_address}

BANK RELATIONSHIP
----------------
Existing Axis Account: {has_axis_account}
Account Number: {account_number}
Account Type: {account_type}
Branch: {branch_name}

CARD PREFERENCES
---------------
Desired Card Type: {card_type}  # My Zone/Neo/Magnus/Privilege
Credit Limit Expected: {desired_limit}
"""

# Kotak Mahindra Bank Credit Card Application Template
kotak_credit_card_template = """
**KOTAK MAHINDRA BANK - CREDIT CARD APPLICATION**
Application ID: {application_id}

APPLICANT DETAILS
----------------
Name as Desired on Card: {card_name}
Full Name: {full_name}
Date of Birth: {dob}
PAN Number: {pan}
Aadhaar Number: {aadhaar}

CONTACT DETAILS
--------------
Mobile Number: {mobile}
Landline: {landline}
Email ID: {email}
Correspondence Address: {corr_address}
Permanent Address: {perm_address}

OCCUPATION DETAILS
-----------------
Nature of Occupation: {occupation}
Company Name: {company}
Current Role: {designation}
Work Experience: {total_experience}
Annual Income: {gross_income}

BANKING DETAILS
--------------
Existing Kotak Account: {is_kotak_customer}
Account Number: {account_number}
Relationship Period: {relationship_duration}

CARD SELECTION
-------------
Card Variant: {card_variant}  # Royale/League/Dream Different
Priority Pass Required: {priority_pass}
"""

# Punjab National Bank (PNB) Credit Card Application Template
pnb_credit_card_template = """
**PUNJAB NATIONAL BANK - CREDIT CARD APPLICATION**
Application Number: {app_number}
Branch Name: {branch_name}

APPLICANT DETAILS
----------------
Full Name: {full_name}
Father's/Spouse Name: {guardian_name}
Date of Birth: {dob}
Gender: {gender}
Nationality: {nationality}

IDENTIFICATION & DOCUMENTS
------------------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Form 60 (If no PAN): {form_60}  # Yes/No

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Residence Phone: {residence_phone}
Email Address: {email}
Current Address: {current_address}
City: {city}
State: {state}
PIN Code: {pin}

EMPLOYMENT DETAILS
----------------
Occupation: {occupation}  # Service/Business/Self-Employed/Others
Employer Name: {employer}
Office Address: {office_address}
Annual Income: {annual_income}

BANKING DETAILS
-------------
PNB Account Number: {account_number}
Account Type: {account_type}
Customer ID: {customer_id}

CARD DETAILS
-----------
Card Type: {card_type}  # Premium/Gold/Platinum
"""

# Yes Bank Credit Card Application Template
yes_bank_credit_card_template = """
**YES BANK - CREDIT CARD APPLICATION**
Reference ID: {ref_id}

PERSONAL INFORMATION
------------------
Name: {name}
Date of Birth: {dob}
Mother's Name: {mother_name}
Marital Status: {marital_status}
Education: {education}

KYC DETAILS
----------
PAN Card Number: {pan}
Aadhaar Number: {aadhaar}
Passport Number (Optional): {passport}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Email ID: {email}
Residence Address: {residence_address}
City: {city}
PIN Code: {pin}
Years at Current Address: {years_at_residence}

PROFESSIONAL INFORMATION
----------------------
Employment Type: {employment_type}
Company Name: {company_name}
Designation: {designation}
Work Experience: {work_exp}
Monthly Income: {monthly_income}

BANKING RELATIONSHIP
------------------
Existing Yes Bank Account: {is_yes_customer}  # Yes/No
Account Number: {account_number}
Savings/Current: {account_type}

CARD PREFERENCE
-------------
Card Variant: {card_variant}  # Yes Prosperity/Yes First/Yes Premia
Annual Fee Option: {annual_fee_option}  # Standard/Lifetime Free
"""

# Canara Bank Credit Card Application Template
canara_credit_card_template = """
**CANARA BANK - CREDIT CARD APPLICATION**
Application Number: {application_no}
Branch: {branch}

PERSONAL PARTICULARS
------------------
Applicant Name: {applicant_name}
Date of Birth: {dob}
Father's/Spouse Name: {guardian_name}
Category: {category}  # General/SC/ST/OBC

IDENTIFICATION
------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}

ADDRESS DETAILS
-------------
Present Address: {present_address}
Permanent Address: {permanent_address}
Mobile Number: {mobile}
Email: {email}

INCOME DETAILS
------------
Occupation: {occupation}
Organization: {organization}
Monthly Income: {monthly_income}
Other Income: {other_income}

BANK ACCOUNT DETAILS
------------------
Account Number: {account_number}
Account Type: {account_type}
Branch IFSC: {ifsc_code}
Customer ID: {customer_id}

CARD REQUIREMENTS
---------------
Card Type: {card_type}  # RuPay/Visa/MasterCard
Variant: {variant}  # Select/Gold/Platinum
"""

# Union Bank Credit Card Application Template
union_bank_credit_card_template = """
**UNION BANK OF INDIA - CREDIT CARD APPLICATION**
Application Reference: {app_ref}

PERSONAL DETAILS
--------------
Name: {name}
Date of Birth: {dob}
Gender: {gender}
Nationality: {nationality}
Category: {category}  # General/SC/ST/OBC/Minority

IDENTITY PROOF
------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Other ID Type: {other_id_type}
Other ID Number: {other_id_number}

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Alternate Number: {alternate_mobile}
Email Address: {email}
Residential Address: {residential_address}
PIN Code: {pin}

EMPLOYMENT & INCOME
----------------
Employment Type: {employment_type}
Employer Name: {employer}
Office Address: {office_address}
Monthly Income: {monthly_income}
Other Income Sources: {other_income}

EXISTING RELATIONSHIPS
-------------------
Union Bank Account: {has_union_account}
Account Number: {account_number}
Home Branch: {home_branch}
IFSC Code: {ifsc}

CARD SELECTION
------------
Card Type: {card_type}  # Classic/Gold/Platinum/Signature
Add-on Card Required: {addon_required}  # Yes/No
"""

# Bank of Baroda Credit Card Application Template
bob_credit_card_template = """
**BANK OF BARODA - CREDIT CARD APPLICATION**
Reference Number: {ref_number}

APPLICANT DETAILS
---------------
Full Name: {full_name}
Date of Birth: {dob}
Mother's Name: {mother_name}
Father's Name: {father_name}
Marital Status: {marital_status}

IDENTIFICATION
------------
PAN Card: {pan}
Aadhaar Number: {aadhaar}
Driving License: {driving_license}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Landline: {landline}
Email ID: {email}
Residential Address: {res_address}
Office Address: {office_address}

OCCUPATION DETAILS
----------------
Employment Status: {employment_status}
Company Name: {company}
Designation: {designation}
Years in Current Job: {years_in_job}
Gross Annual Income: {annual_income}

BANK RELATIONSHIP
--------------
BoB Account Holder: {is_bob_customer}
Account Number: {account_number}
Branch Name: {branch_name}
Customer ID: {customer_id}

CARD REQUIREMENTS
--------------
Card Variant: {card_variant}  # Easy/Select/Premier
Insurance Required: {insurance_required}  # Yes/No
SMS Alerts: {sms_alerts}  # Yes/No
"""