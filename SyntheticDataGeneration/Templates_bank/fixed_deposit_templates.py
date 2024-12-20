# SBI Fixed Deposit Template
sbi_fd_template = """
**STATE BANK OF INDIA - FIXED DEPOSIT APPLICATION**
Reference Number: {ref_number}
Date: {date}

APPLICANT DETAILS
---------------
Name: {applicant_name}
Customer ID: {customer_id}
PAN Number: {pan}
Date of Birth: {dob}
Mobile Number: {mobile}
Email ID: {email}

DEPOSIT DETAILS
-------------
Deposit Amount: {deposit_amount}
Deposit Period: {deposit_period}  # Years/Months/Days
Interest Rate: {interest_rate}
Deposit Type: {deposit_type}  # Regular/Tax Saving/Senior Citizen
Interest Payout: {interest_payout}  # Monthly/Quarterly/Annual/Cumulative

ACCOUNT DETAILS
-------------
Source Account: {source_account}
Branch: {branch_name}
Mode of Operation: {operation_mode}  # Single/Joint/Either or Survivor

MATURITY INSTRUCTIONS
------------------
Maturity Proceeds: {maturity_instruction}  # Auto Renew/Credit to Account
Auto Renewal Period: {renewal_period}
Credit Account: {credit_account}

NOMINATION DETAILS
---------------
Nominee Name: {nominee_name}
Relationship: {nominee_relation}
Nominee DOB: {nominee_dob}
"""

# HDFC Fixed Deposit Template
hdfc_fd_template = """
**HDFC BANK - FIXED DEPOSIT APPLICATION**
Application ID: {application_id}

PERSONAL INFORMATION
-----------------
Full Name: {full_name}
Customer ID: {cust_id}
PAN Number: {pan}
Mobile Number: {mobile}
Email Address: {email}
Category: {category}  # Regular/Senior Citizen/Staff

DEPOSIT SCHEME
------------
Deposit Type: {deposit_type}  # Regular/Tax Saver/Super Saver
Amount: {amount}
Tenure: {tenure}
Interest Rate Applicable: {interest_rate}
Interest Payment Frequency: {interest_frequency}

FUNDING DETAILS
-------------
Debit Account Number: {debit_account}
Payment Mode: {payment_mode}  # Account Transfer/Cheque/NEFT
Branch: {branch}

MATURITY INSTRUCTIONS
------------------
Maturity Action: {maturity_action}  # Renew Principal/Renew with Interest/Pay
Credit Account: {credit_account}
Renewal Period: {renewal_period}

NOMINATION
---------
Nominee Name: {nominee_name}
Relationship: {relationship}
Date of Birth: {nominee_dob}
Address: {nominee_address}
"""

# ICICI Fixed Deposit Template
icici_fd_template = """
**ICICI BANK - FIXED DEPOSIT APPLICATION**
Reference Number: {ref_no}

CUSTOMER INFORMATION
-----------------
Name: {customer_name}
Customer ID: {customer_id}
PAN: {pan_number}
Mobile: {mobile}
Email: {email}
Category: {category}  # General/Senior Citizen

DEPOSIT REQUIREMENTS
-----------------
Deposit Amount: {deposit_amount}
Deposit Period: {deposit_period}
Interest Rate: {interest_rate}
Deposit Type: {deposit_type}  # Cumulative/Non-Cumulative
Interest Payout Option: {payout_option}

SOURCE OF FUNDS
-------------
Account Number: {source_account}
Branch Name: {branch_name}
Mode of Funding: {funding_mode}

MATURITY DETAILS
--------------
Maturity Instructions: {maturity_instruction}
Auto Renewal Terms: {renewal_terms}
Payout Account: {payout_account}

NOMINEE INFORMATION
----------------
Nominee Name: {nominee_name}
Relationship: {relationship}
Date of Birth: {nominee_dob}
Guardian Details: {guardian_details}  # If nominee is minor
"""

# Axis Bank Fixed Deposit Template
axis_fd_template = """
**AXIS BANK - FIXED DEPOSIT APPLICATION**
Form Number: {form_number}

APPLICANT DETAILS
--------------
Name: {name}
Customer ID: {customer_id}
PAN Number: {pan}
Mobile Number: {mobile}
Email ID: {email}
Customer Type: {customer_type}  # Regular/Senior Citizen/Staff

DEPOSIT DETAILS
-------------
Scheme Type: {scheme_type}
Principal Amount: {principal_amount}
Tenure: {tenure}
Interest Rate: {interest_rate}
Interest Payout: {interest_payout}

FUNDING INFORMATION
----------------
Source Account: {source_account}
Mode of Funding: {funding_mode}
Branch: {branch}

MATURITY INSTRUCTIONS
------------------
Maturity Action: {maturity_action}
Renewal Instructions: {renewal_instructions}
Credit Account: {credit_account}

NOMINATION DETAILS
---------------
Nominee Name: {nominee_name}
Relationship: {relationship}
DOB: {nominee_dob}
Guardian Name: {guardian_name}  # If applicable
"""

# Kotak Mahindra Bank Fixed Deposit Template
kotak_fd_template = """
**KOTAK MAHINDRA BANK - FIXED DEPOSIT APPLICATION**
Application ID: {application_id}

PERSONAL DETAILS
-------------
Account Holder Name: {holder_name}
Customer Relationship Number: {crn}
PAN: {pan}
Mobile: {mobile}
Email: {email}
Category: {category}

DEPOSIT SELECTION
--------------
Deposit Type: {deposit_type}
Amount: {amount}
Period: {period}
Interest Rate: {interest_rate}
Interest Payment: {interest_payment}

FUNDING DETAILS
-------------
Debit Account: {debit_account}
Mode of Funding: {funding_mode}
Branch: {branch}

MATURITY PROFILE
-------------
Maturity Instructions: {maturity_instructions}
Renewal Terms: {renewal_terms}
Settlement Account: {settlement_account}

NOMINEE DETAILS
-------------
Nominee Name: {nominee_name}
Relationship: {relationship}
Date of Birth: {nominee_dob}
"""

# PNB Fixed Deposit Template
pnb_fd_template = """
**PUNJAB NATIONAL BANK - FIXED DEPOSIT APPLICATION**
Reference Number: {ref_number}

DEPOSITOR INFORMATION
------------------
Name: {depositor_name}
Customer ID: {customer_id}
PAN Number: {pan}
Mobile: {mobile}
Email: {email}
Category: {category}

DEPOSIT SCHEME
------------
Scheme Name: {scheme_name}
Deposit Amount: {deposit_amount}
Period: {deposit_period}
Rate of Interest: {interest_rate}
Interest Frequency: {interest_frequency}

ACCOUNT DETAILS
-------------
Source Account: {source_account}
Branch Name: {branch_name}
Mode of Operation: {operation_mode}

MATURITY DETAILS
-------------
Maturity Instructions: {maturity_instructions}
Auto Renewal: {auto_renewal}
Credit Account: {credit_account}

NOMINATION
---------
Nominee Name: {nominee_name}
Relationship: {relationship}
Nominee DOB: {nominee_dob}
"""

# Yes Bank Fixed Deposit Template
yes_bank_fd_template = """
**YES BANK - FIXED DEPOSIT APPLICATION**
Form Number: {form_number}

CUSTOMER DETAILS
-------------
Customer Name: {customer_name}
Customer ID: {customer_id}
PAN: {pan}
Contact Number: {contact_number}
Email ID: {email_id}
Category: {category}

DEPOSIT INFORMATION
----------------
Deposit Type: {deposit_type}
Amount: {amount}
Tenure: {tenure}
Interest Rate: {interest_rate}
Interest Payout: {interest_payout}

FUNDING DETAILS
-------------
Source Account: {source_account}
Mode of Funding: {funding_mode}
Branch Location: {branch_location}

MATURITY INSTRUCTIONS
------------------
Maturity Action: {maturity_action}
Renewal Instructions: {renewal_instructions}
Settlement Account: {settlement_account}

NOMINEE INFORMATION
----------------
Nominee Name: {nominee_name}
Relationship: {relationship}
Date of Birth: {nominee_dob}
"""

# Canara Bank Fixed Deposit Template
canara_fd_template = """
**CANARA BANK - FIXED DEPOSIT APPLICATION**
Application Number: {application_no}

APPLICANT DETAILS
--------------
Name: {name}
Customer ID: {customer_id}
PAN Number: {pan}
Mobile Number: {mobile}
Email ID: {email}
Category: {category}

DEPOSIT DETAILS
-------------
Scheme Type: {scheme_type}
Deposit Amount: {deposit_amount}
Period: {period}
Interest Rate: {interest_rate}
Interest Payment: {interest_payment}

ACCOUNT INFORMATION
----------------
Source Account: {source_account}
Branch: {branch}
Mode of Operation: {operation_mode}

MATURITY INSTRUCTIONS
------------------
Maturity Action: {maturity_action}
Renewal Terms: {renewal_terms}
Credit Account: {credit_account}

NOMINATION DETAILS
---------------
Nominee Name: {nominee_name}
Relationship: {relationship}
Nominee DOB: {nominee_dob}
"""

# Union Bank Fixed Deposit Template
union_bank_fd_template = """
**UNION BANK OF INDIA - FIXED DEPOSIT APPLICATION**
Reference Number: {ref_no}

CUSTOMER DETAILS
-------------
Full Name: {full_name}
Customer ID: {customer_id}
PAN Number: {pan}
Mobile: {mobile}
Email: {email}
Category: {category}

DEPOSIT SCHEME
------------
Deposit Type: {deposit_type}
Amount: {amount}
Tenure: {tenure}
Interest Rate: {interest_rate}
Interest Frequency: {interest_frequency}

FUNDING DETAILS
-------------
Debit Account: {debit_account}
Branch Name: {branch_name}
Mode of Funding: {funding_mode}

MATURITY PROFILE
-------------
Maturity Instructions: {maturity_instructions}
Auto Renewal: {auto_renewal}
Settlement Account: {settlement_account}

NOMINEE DETAILS
-------------
Nominee Name: {nominee_name}
Relationship: {relationship}
Date of Birth: {nominee_dob}
Guardian Details: {guardian_details}
"""

# Bank of Baroda Fixed Deposit Template
bob_fd_template = """
**BANK OF BARODA - FIXED DEPOSIT APPLICATION**
Form Number: {form_number}

PERSONAL INFORMATION
-----------------
Customer Name: {customer_name}
Customer ID: {customer_id}
PAN: {pan}
Mobile Number: {mobile}
Email ID: {email}
Category: {category}

DEPOSIT DETAILS
-------------
Scheme Selected: {scheme}
Deposit Amount: {deposit_amount}
Period: {period}
Interest Rate: {interest_rate}
Interest Payout: {interest_payout}

FUNDING INFORMATION
----------------
Source Account: {source_account}
Branch: {branch}
Mode of Funding: {funding_mode}

MATURITY INSTRUCTIONS
------------------
Maturity Action: {maturity_action}
Renewal Instructions: {renewal_instructions}
Credit Account: {credit_account}

NOMINATION
---------
Nominee Name: {nominee_name}
Relationship: {relationship}
Date of Birth: {nominee_dob}
Guardian Name: {guardian_name}
""" 