# SBI Loan Application Template
sbi_loan_template = """
**STATE BANK OF INDIA - LOAN APPLICATION**
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
Years at Current Address: {years_at_residence}

EMPLOYMENT DETAILS
-----------------
Occupation Type: {occupation}  # Salaried/Self-Employed/Business
Company Name: {company_name}
Annual Income: {annual_income}
Years in Current Job: {work_experience}

LOAN DETAILS
-----------
Loan Type: {loan_type}  # Home/Personal/Vehicle/Education
Loan Amount Required: {loan_amount}
Loan Tenure (Years): {loan_tenure}
Purpose of Loan: {loan_purpose}

EXISTING LOANS
-------------
Any Existing Loans: {has_existing_loans}  # Yes/No
Total EMI Amount: {current_emi}
"""

# HDFC Loan Application Template
hdfc_loan_template = """
**HDFC BANK - LOAN APPLICATION**
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
Residence Type: {residence_type}  # Self-Owned/Rented/Parental
Years at Current Address: {years_at_address}

PROFESSIONAL DETAILS
-------------------
Employment Type: {employment_type}
Employer Name: {employer}
Monthly Income: {monthly_income}
Work Experience: {total_experience}
Office Address: {office_address}

LOAN REQUIREMENTS
----------------
Loan Product: {loan_product}  # Personal/Home/Car/Education
Required Amount: {required_amount}
Tenure Requested: {tenure_months}
Purpose: {purpose}

FINANCIAL DETAILS
----------------
Existing EMIs: {existing_emis}
Other Income Sources: {other_income}
Bank Account Number: {account_number}
"""

# ICICI Loan Application Template
icici_loan_template = """
**ICICI BANK - LOAN APPLICATION FORM**
Application Number: {app_number}
Date: {application_date}

1. PERSONAL INFORMATION
----------------------
Name: {applicant_name}
Date of Birth: {date_of_birth}
Gender: {gender}
PAN: {pan_number}
Aadhaar: {aadhaar_number}
Contact Information:
  - Phone: {phone_number}
  - Email: {email_address}

2. ADDRESS DETAILS
-----------------
Present Address: {current_address}
City: {city_name}
State: {state}
PIN: {pincode}
Ownership Status: {ownership}  # Own/Rented/Lease
Duration of Stay: {stay_duration}

3. EMPLOYMENT & INCOME
---------------------
Employment Type: {emp_type}  # Salaried/Self-Employed/Business Owner
Organization: {organization_name}
Designation: {designation}
Work Experience: {years_of_experience}
Gross Annual Income: {gross_income}
Net Monthly Income: {net_income}

4. LOAN REQUEST
--------------
Loan Category: {loan_category}  # Personal/Home/Auto/Business
Amount Requested: {requested_amount}
Tenure Required: {tenure}
Purpose: {loan_purpose}

5. BANKING RELATIONSHIPS
-----------------------
Primary Bank: {primary_bank}
Account Type: {account_type}
Account Number: {account_number}
Existing Loans: {existing_loans}  # Yes/No
Current EMIs: {monthly_emi}
"""

# Axis Bank Loan Application Template
axis_loan_template = """
**AXIS BANK LOAN APPLICATION**
Reference ID: {reference_id}

APPLICANT DETAILS
----------------
Full Name: {full_name}
DOB: {dob}
Gender: {gender}
Marital Status: {marital_status}
Identity Details:
  PAN: {pan}
  Aadhaar: {aadhaar}
Contact:
  Mobile: {mobile_no}
  Email: {email_id}

RESIDENCE INFORMATION
-------------------
Current Address: {residence_address}
Landmark: {landmark}
City & State: {city_state}
PIN Code: {pin_code}
Residence Type: {residence_type}  # Owned/Rented/Family
Stay Duration: {duration_years}

OCCUPATION DETAILS
----------------
Profession: {profession}  # Service/Business/Self-Employed
Company/Business Name: {company_name}
Industry Type: {industry}
Years in Current Work: {work_years}
Monthly Income: {monthly_income}
Annual Turnover: {annual_turnover}  # For Business/Self-Employed

LOAN REQUIREMENTS
---------------
Loan Type: {loan_type}
Required Amount: {loan_amount}
Tenure (Months): {tenure}
Repayment Mode: {repayment_mode}
Purpose: {purpose}

FINANCIAL HISTORY
---------------
Existing Credit Facilities: {existing_credits}
Monthly Obligations: {monthly_obligations}
Relationship with Axis Bank: {relationship_period}
"""

# Kotak Mahindra Bank Loan Application Template
kotak_loan_template = """
**KOTAK MAHINDRA BANK - LOAN APPLICATION**
Application ID: {application_id}
Branch: {branch_name}

PERSONAL PARTICULARS
------------------
Applicant Name: {name}
Date of Birth: {birth_date}
Gender: {gender}
Identification:
  - PAN Number: {pan_no}
  - Aadhaar Number: {aadhaar_no}
Contact Details:
  - Mobile: {mobile_number}
  - Email: {email_id}
  - Alternative Contact: {alt_contact}

RESIDENTIAL DETAILS
-----------------
Present Address: {current_address}
City: {city}
State: {state}
PIN: {pin_code}
Residence Status: {residence_status}  # Self-Owned/Rented/Parental
Years of Stay: {years_at_address}

EMPLOYMENT INFORMATION
--------------------
Occupation: {occupation_type}  # Salaried/Self-Employed Professional/Business
Employer/Business: {employer_name}
Industry Sector: {industry_type}
Work Experience: {total_experience}
Income Details:
  - Monthly Income: {monthly_income}
  - Other Income: {other_income}
  - Annual Business Turnover: {business_turnover}  # If applicable

LOAN PARTICULARS
--------------
Product Type: {loan_product}  # Personal/Home/Business/Vehicle
Requested Amount: {request_amount}
Tenure Requested: {tenure_months}
Purpose of Loan: {purpose}
Preferred EMI Date: {preferred_emi_date}

BANKING RELATIONSHIP
------------------
Existing Customer: {existing_customer}  # Yes/No
Account Number: {account_no}
Current Banking Products: {current_products}
Outstanding Loans: {outstanding_loans}
Total EMI Load: {total_emi}
"""

# Punjab National Bank (PNB) Loan Application Template
pnb_loan_template = """
**PUNJAB NATIONAL BANK - LOAN APPLICATION**
Application Number: {app_number}
Branch: {branch_name}

PERSONAL INFORMATION
------------------
Applicant Name: {full_name}
Father's/Spouse Name: {guardian_name}
Date of Birth: {dob}
Gender: {gender}
Category: {category}  # General/SC/ST/OBC
Nationality: {nationality}

IDENTIFICATION DETAILS
-------------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Driving License: {driving_license}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Landline: {landline}
Email: {email}
Present Address: {present_address}
Permanent Address: {permanent_address}
PIN Code: {pin}

EMPLOYMENT & INCOME
----------------
Occupation: {occupation}
Employer/Business Name: {employer}
Work Experience: {experience}
Monthly Income: {monthly_income}
Other Income: {other_income}
Annual Turnover: {annual_turnover}

LOAN REQUIREMENTS
--------------
Loan Type: {loan_type}  # Housing/Vehicle/Education/Personal
Amount Required: {loan_amount}
Tenure Required: {loan_tenure}
Purpose: {purpose}
Security Offered: {security}

EXISTING LIABILITIES
-----------------
Existing Loans: {existing_loans}
Total EMI: {current_emi}
Other Liabilities: {other_liabilities}
"""

# Yes Bank Loan Application Template
yes_bank_loan_template = """
**YES BANK - LOAN APPLICATION**
Reference Number: {ref_number}

APPLICANT DETAILS
---------------
Name: {applicant_name}
Date of Birth: {dob}
Gender: {gender}
Marital Status: {marital_status}
Education: {education}

KYC INFORMATION
-------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Address Proof Type: {address_proof_type}
Address Proof Number: {address_proof_number}

CONTACT INFORMATION
----------------
Mobile Number: {mobile}
Email ID: {email}
Current Address: {current_address}
Permanent Address: {permanent_address}
Years at Current Address: {years_at_residence}

EMPLOYMENT DETAILS
---------------
Employment Type: {employment_type}
Company/Business Name: {company_name}
Designation: {designation}
Total Experience: {total_experience}
Current Experience: {current_experience}

INCOME DETAILS
------------
Gross Monthly Income: {monthly_income}
Net Monthly Income: {net_income}
Other Income: {other_income}
Income Source: {income_source}

LOAN DETAILS
----------
Product Type: {loan_type}
Required Amount: {required_amount}
Tenure (Months): {tenure}
Purpose of Loan: {purpose}
Collateral Details: {collateral}

EXISTING RELATIONSHIPS
------------------
Existing Customer: {existing_customer}
Products Held: {existing_products}
Account Number: {account_number}
"""

# Canara Bank Loan Application Template
canara_loan_template = """
**CANARA BANK - LOAN APPLICATION**
Application Number: {application_no}
Branch: {branch}

PERSONAL DETAILS
--------------
Full Name: {full_name}
Father's/Spouse Name: {guardian_name}
Date of Birth: {dob}
Age: {age}
Gender: {gender}
Category: {category}  # General/SC/ST/OBC

IDENTITY DETAILS
--------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Driving License: {driving_license}

ADDRESS DETAILS
-------------
Present Address: {present_address}
Permanent Address: {permanent_address}
PIN Code: {pin}
Contact Numbers:
    Mobile: {mobile}
    Residence: {residence_phone}
Email: {email}

OCCUPATION & INCOME
----------------
Nature of Occupation: {occupation}
Employer/Business: {employer}
Annual Income: {annual_income}
Other Income: {other_income}
Total Experience: {experience}

LOAN REQUEST
----------
Type of Loan: {loan_type}
Amount Required: {loan_amount}
Repayment Period: {repayment_period}
Purpose: {purpose}
Security Offered: {security}

BANK ACCOUNT DETAILS
-----------------
Account Number: {account_number}
Account Type: {account_type}
IFSC Code: {ifsc}
Customer ID: {customer_id}
"""

# Union Bank Loan Application Template
union_bank_loan_template = """
**UNION BANK OF INDIA - LOAN APPLICATION**
Reference Number: {ref_number}

APPLICANT INFORMATION
------------------
Name: {name}
Date of Birth: {dob}
Age: {age}
Gender: {gender}
Marital Status: {marital_status}
Category: {category}  # General/SC/ST/OBC

IDENTIFICATION
------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Alternative Number: {alt_mobile}
Email: {email}
Residential Address: {res_address}
Office Address: {office_address}

EMPLOYMENT DETAILS
---------------
Occupation: {occupation}
Employer/Business Name: {employer}
Years in Current Job/Business: {current_experience}
Total Experience: {total_experience}

INCOME DETAILS
------------
Gross Monthly Income: {monthly_income}
Annual Income: {annual_income}
Other Income Sources: {other_income}
ITR for Last 2 Years: {itr_details}

LOAN REQUIREMENTS
--------------
Loan Scheme: {loan_scheme}
Amount Required: {amount}
Tenure Required: {tenure}
Purpose: {purpose}
Security Details: {security}

EXISTING FACILITIES
----------------
Existing Loans: {existing_loans}
Total EMI: {total_emi}
Credit Card Dues: {card_dues}
"""

# Bank of Baroda Loan Application Template
bob_loan_template = """
**BANK OF BARODA - LOAN APPLICATION**
Application Number: {application_no}

PERSONAL DETAILS
--------------
Applicant Name: {applicant_name}
Father's/Spouse Name: {guardian_name}
Date of Birth: {dob}
Gender: {gender}
Marital Status: {marital_status}
Category: {category}

IDENTITY VERIFICATION
------------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Passport Number: {passport}

CONTACT INFORMATION
----------------
Mobile Number: {mobile}
Landline Number: {landline}
Email Address: {email}
Current Address: {current_address}
Permanent Address: {permanent_address}

EMPLOYMENT INFORMATION
-------------------
Occupation Type: {occupation_type}
Employer/Business Name: {employer}
Designation: {designation}
Work Experience: {experience}
Business/Office Address: {work_address}

FINANCIAL DETAILS
--------------
Monthly Income: {monthly_income}
Annual Income: {annual_income}
Other Income: {other_income}
Income Source: {income_source}

LOAN DETAILS
----------
Loan Type: {loan_type}
Amount Required: {loan_amount}
Tenure (Months): {tenure}
Purpose of Loan: {purpose}
Security Offered: {security}

BANKING RELATIONSHIP
-----------------
Existing Customer: {existing_customer}
Account Number: {account_number}
Branch Name: {branch_name}
Other Banking Relations: {other_banks}
""" 