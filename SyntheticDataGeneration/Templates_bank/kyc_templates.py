# SBI KYC Application Template
sbi_kyc_template = """
**STATE BANK OF INDIA - KYC FORM**
Reference Number: {ref_number}
Date: {date}

PERSONAL INFORMATION
------------------
Customer Name: {customer_name}
Father's/Spouse Name: {guardian_name}
Mother's Name: {mother_name}
Date of Birth: {dob}
Gender: {gender}
Marital Status: {marital_status}
Nationality: {nationality}

IDENTITY DOCUMENTS
----------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Passport Number: {passport_no}
Driving License: {driving_license}
Voter ID: {voter_id}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Landline: {landline}
Email ID: {email}

ADDRESS DETAILS
-------------
Current Address: {current_address}
Permanent Address: {permanent_address}
City: {city}
State: {state}
PIN Code: {pin}
Residence Type: {residence_type}  # Owned/Rented/Leased

OCCUPATION DETAILS
----------------
Occupation Type: {occupation}
Annual Income Range: {income_range}
Source of Income: {income_source}
Employer/Business Name: {employer}

ACCOUNT DETAILS
-------------
Account Type: {account_type}
Account Number: {account_number}
Branch Code: {branch_code}
"""

# HDFC KYC Application Template
hdfc_kyc_template = """
**HDFC BANK - KYC APPLICATION**
Application ID: {application_id}

CUSTOMER DETAILS
--------------
Full Name: {full_name}
Date of Birth: {dob}
Gender: {gender}
Citizenship: {citizenship}
Tax Residency: {tax_residency}

PROOF OF IDENTITY
---------------
PAN Card Number: {pan}
Aadhaar Number: {aadhaar}
Form 60 (if no PAN): {form_60}
Additional ID Type: {other_id_type}
Additional ID Number: {other_id_number}

ADDRESS INFORMATION
-----------------
Residential Address: {residential_address}
Office Address: {office_address}
Correspondence Address: {corr_address}
Address Proof Type: {address_proof}
Proof Number: {proof_number}

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Alternate Number: {alt_mobile}
Email Address: {email}
Landline: {landline}

OCCUPATION & INCOME
----------------
Occupation: {occupation}
Nature of Business: {business_nature}
Annual Income: {annual_income}
Source of Funds: {fund_source}

FATCA DECLARATION
---------------
US Person Status: {us_person}
Tax Residency Country: {tax_country}
Tax ID Number: {tax_id}
"""

# ICICI KYC Application Template
icici_kyc_template = """
**ICICI BANK - KYC FORM**
Reference Number: {ref_no}

BASIC INFORMATION
---------------
Customer Name: {customer_name}
Date of Birth: {dob}
Place of Birth: {birth_place}
Gender: {gender}
Marital Status: {marital_status}
Nationality: {nationality}

IDENTITY VERIFICATION
------------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Passport Details: {passport_details}
Other Valid ID: {other_id}
Photo Submission: {photo_submitted}

CONTACT & ADDRESS
---------------
Mobile Number: {mobile}
Email ID: {email}
Residence Phone: {residence_phone}
Current Address: {current_address}
Permanent Address: {permanent_address}
Address Proof Type: {address_proof}

OCCUPATION DETAILS
----------------
Employment Type: {employment_type}
Employer Name: {employer}
Office Address: {office_address}
Annual Income: {annual_income}
Nature of Business: {business_nature}

ADDITIONAL DETAILS
----------------
Purpose of Account: {account_purpose}
Expected Transactions: {expected_transactions}
Risk Category: {risk_category}
"""

# Axis Bank KYC Application Template
axis_kyc_template = """
**AXIS BANK - KYC APPLICATION**
Form Number: {form_number}

PERSONAL DETAILS
--------------
Name: {name}
Date of Birth: {dob}
Gender: {gender}
Nationality: {nationality}
Resident Status: {resident_status}

IDENTIFICATION
------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Passport Number: {passport}
Other ID Type: {other_id_type}
Other ID Number: {other_id_number}

ADDRESS DETAILS
-------------
Residential Address: {res_address}
Office Address: {off_address}
Preferred Mailing: {mailing_preference}
Proof Submitted: {address_proof}

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Email Address: {email}
Alternate Contact: {alt_contact}

OCCUPATION & INCOME
----------------
Occupation Type: {occupation}
Industry: {industry}
Annual Income: {annual_income}
Source of Wealth: {wealth_source}

ACCOUNT RELATED
-------------
Account Type: {account_type}
Banking Purpose: {banking_purpose}
Expected Turnover: {expected_turnover}
"""

# Kotak Mahindra Bank KYC Template
kotak_kyc_template = """
**KOTAK MAHINDRA BANK - KYC FORM**
Application ID: {application_id}

CUSTOMER INFORMATION
-----------------
Full Name: {full_name}
Date of Birth: {dob}
Place of Birth: {birth_place}
Mother's Name: {mother_name}
Father's Name: {father_name}

IDENTITY DETAILS
--------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Driving License: {driving_license}

CONTACT & ADDRESS
---------------
Mobile Number: {mobile}
Email ID: {email}
Current Address: {current_address}
Permanent Address: {permanent_address}
Residence Type: {residence_type}

OCCUPATION DETAILS
----------------
Occupation: {occupation}
Employer/Business: {employer}
Work Address: {work_address}
Annual Income: {annual_income}

FATCA/CRS DETAILS
---------------
Tax Residency: {tax_residency}
FATCA Status: {fatca_status}
CRS Declaration: {crs_declaration}
"""

# PNB KYC Application Template
pnb_kyc_template = """
**PUNJAB NATIONAL BANK - KYC FORM**
Reference Number: {ref_number}

PERSONAL DETAILS
--------------
Applicant Name: {applicant_name}
Parent/Spouse Name: {guardian_name}
Date of Birth: {dob}
Gender: {gender}
Category: {category}  # General/SC/ST/OBC

PROOF OF IDENTITY
---------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Passport Number: {passport}

ADDRESS INFORMATION
----------------
Present Address: {present_address}
Permanent Address: {permanent_address}
Address Proof Type: {proof_type}
Proof Number: {proof_number}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Landline Number: {landline}
Email Address: {email}

OCCUPATION & INCOME
----------------
Occupation: {occupation}
Annual Income: {annual_income}
Source of Income: {income_source}
"""

# Yes Bank KYC Application Template
yes_bank_kyc_template = """
**YES BANK - KYC APPLICATION**
Form Number: {form_number}

PERSONAL INFORMATION
-----------------
Customer Name: {customer_name}
Date of Birth: {dob}
Gender: {gender}
Nationality: {nationality}
Resident Status: {resident_status}

IDENTIFICATION DETAILS
-------------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Other ID Type: {other_id_type}
Other ID Number: {other_id_number}

ADDRESS DETAILS
-------------
Current Address: {current_address}
Permanent Address: {permanent_address}
Proof Submitted: {address_proof}

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Email ID: {email}
Alternate Contact: {alt_contact}

OCCUPATION DETAILS
----------------
Occupation Type: {occupation}
Employer Details: {employer}
Annual Income: {annual_income}
"""

# Canara Bank KYC Application Template
canara_kyc_template = """
**CANARA BANK - KYC FORM**
Application Number: {application_no}

PERSONAL DETAILS
--------------
Name: {name}
Date of Birth: {dob}
Gender: {gender}
Marital Status: {marital_status}
Category: {category}

IDENTITY PROOF
------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Other ID: {other_id}

ADDRESS DETAILS
-------------
Present Address: {present_address}
Permanent Address: {permanent_address}
Proof Type: {proof_type}

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Landline: {landline}
Email: {email}

OCCUPATION & INCOME
----------------
Occupation: {occupation}
Annual Income: {annual_income}
Income Source: {income_source}
"""

# Union Bank KYC Application Template
union_bank_kyc_template = """
**UNION BANK OF INDIA - KYC FORM**
Reference Number: {ref_no}

CUSTOMER DETAILS
--------------
Full Name: {full_name}
Date of Birth: {dob}
Gender: {gender}
Nationality: {nationality}
Category: {category}

IDENTITY DOCUMENTS
---------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Voter ID: {voter_id}
Other ID: {other_id}

ADDRESS INFORMATION
----------------
Current Address: {current_address}
Permanent Address: {permanent_address}
Address Proof: {address_proof}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Email ID: {email}
Alternate Number: {alt_number}

OCCUPATION DETAILS
---------------
Occupation Type: {occupation}
Annual Income: {annual_income}
Source of Funds: {fund_source}
"""

# Bank of Baroda KYC Application Template
bob_kyc_template = """
**BANK OF BARODA - KYC FORM**
Form Number: {form_number}

PERSONAL INFORMATION
-----------------
Customer Name: {customer_name}
Father's/Spouse Name: {guardian_name}
Date of Birth: {dob}
Gender: {gender}
Marital Status: {marital_status}

PROOF OF IDENTITY
---------------
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Passport Number: {passport}
Other ID Type: {other_id_type}

ADDRESS DETAILS
-------------
Residential Address: {res_address}
Office Address: {off_address}
Correspondence Address: {corr_address}
Proof Submitted: {address_proof}

CONTACT INFORMATION
-----------------
Mobile Number: {mobile}
Email Address: {email}
Landline Number: {landline}

OCCUPATION & INCOME
----------------
Occupation: {occupation}
Annual Income: {annual_income}
Source of Income: {income_source}
Employer Details: {employer}
""" 