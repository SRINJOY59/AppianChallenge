# SBI Account Closure Template
sbi_closure_template = """
**STATE BANK OF INDIA - ACCOUNT CLOSURE REQUEST**
Reference Number: {ref_number}
Date: {date}

ACCOUNT HOLDER DETAILS
--------------------
Name: {account_holder_name}
Customer ID: {customer_id}
Mobile Number: {mobile}
Email ID: {email}

ACCOUNT DETAILS
-------------
Account Number: {account_number}
Account Type: {account_type}
Branch Name: {branch_name}
IFSC Code: {ifsc_code}

CLOSURE DETAILS
-------------
Reason for Closure: {closure_reason}
Mode of Balance Settlement: {settlement_mode}  # Cash/Transfer/DD
Settlement Account Details (if applicable):
    Bank Name: {transfer_bank_name}
    Account Number: {transfer_account_number}
    IFSC Code: {transfer_ifsc}

LINKED SERVICES
-------------
Debit Card Number: {debit_card}
Internet Banking: {internet_banking}  # Active/Inactive
Mobile Banking: {mobile_banking}  # Active/Inactive
Standing Instructions: {standing_instructions}  # Yes/No

DECLARATION
----------
Outstanding Charges: {outstanding_charges}  # Yes/No
Unused Cheque Leaves: {unused_cheques}
"""

# HDFC Account Closure Template
hdfc_closure_template = """
**HDFC BANK - ACCOUNT CLOSURE APPLICATION**
Application ID: {application_id}

PRIMARY ACCOUNT HOLDER
-------------------
Full Name: {full_name}
Customer ID: {cust_id}
Contact Number: {contact_number}
Email Address: {email}

ACCOUNT INFORMATION
-----------------
Account Number: {account_number}
Account Type: {account_type}
Home Branch: {home_branch}
Account Opening Date: {opening_date}

CLOSURE REQUEST
-------------
Primary Reason: {primary_reason}
Detailed Reason: {detailed_reason}
Preferred Closure Date: {closure_date}

SETTLEMENT INSTRUCTIONS
--------------------
Settlement Method: {settlement_method}  # NEFT/RTGS/DD/Cash
Beneficiary Details:
    Name: {beneficiary_name}
    Account Number: {beneficiary_account}
    Bank & Branch: {beneficiary_bank}
    IFSC Code: {beneficiary_ifsc}

PRODUCT SURRENDERS
---------------
Debit Card: {debit_card_number}
Credit Card Linked: {credit_card_linked}  # Yes/No
Investment Accounts: {investment_accounts}  # Yes/No
Locker Facility: {locker_facility}  # Yes/No
"""

# ICICI Account Closure Template
icici_closure_template = """
**ICICI BANK - ACCOUNT CLOSURE FORM**
Reference Number: {ref_no}

CUSTOMER DETAILS
--------------
Customer Name: {customer_name}
Customer ID: {customer_id}
Mobile Number: {mobile}
Email ID: {email}

ACCOUNT DETAILS
-------------
Account Number: {account_number}
Account Type: {account_type}
Branch: {branch_name}
Opening Date: {opening_date}

CLOSURE INFORMATION
----------------
Reason for Closure: {closure_reason}
Closure Date Requested: {requested_date}
Balance Settlement Mode: {settlement_mode}
Transfer Details (if applicable):
    Bank Name: {transfer_bank}
    Account Number: {transfer_account}
    IFSC Code: {transfer_ifsc}

LINKED FACILITIES
--------------
Debit Card Status: {debit_card_status}
Internet Banking: {internet_banking}
Bill Payments: {bill_payments}
UPI Services: {upi_services}

DECLARATIONS
----------
Outstanding Dues: {outstanding_dues}
Uncleared Cheques: {uncleared_cheques}
"""

# Axis Bank Account Closure Template
axis_closure_template = """
**AXIS BANK - ACCOUNT CLOSURE REQUEST**
Form Number: {form_number}

ACCOUNT HOLDER INFORMATION
-----------------------
Name: {name}
Customer ID: {customer_id}
Contact Details:
    Mobile: {mobile}
    Email: {email}

ACCOUNT DETAILS
-------------
Account Number: {account_number}
Type of Account: {account_type}
Branch Name: {branch}
IFSC Code: {ifsc}

CLOSURE REQUEST DETAILS
--------------------
Reason for Closure: {reason}
Preferred Closure Date: {preferred_date}
Balance Amount: {balance_amount}
Settlement Instructions: {settlement_instructions}

LINKED PRODUCTS
-------------
Debit Card Number: {debit_card}
Credit Card Linkage: {credit_card}
Investment Account: {investment_account}
Loan Account: {loan_account}

DECLARATIONS
----------
Pending Charges: {pending_charges}
Unused Cheques: {unused_cheques}
Standing Instructions: {standing_instructions}
"""

# Kotak Mahindra Bank Account Closure Template
kotak_closure_template = """
**KOTAK MAHINDRA BANK - ACCOUNT CLOSURE APPLICATION**
Application ID: {application_id}

PERSONAL DETAILS
--------------
Account Holder Name: {holder_name}
Customer Relationship Number: {crn}
Mobile Number: {mobile}
Email Address: {email}

ACCOUNT INFORMATION
----------------
Account Number: {account_number}
Account Type: {account_type}
Branch: {branch}
Account Status: {account_status}

CLOSURE DETAILS
-------------
Reason for Closure: {reason}
Requested Closure Date: {closure_date}
Mode of Settlement: {settlement_mode}
Settlement Details:
    Beneficiary Name: {beneficiary_name}
    Account Number: {beneficiary_account}
    IFSC Code: {beneficiary_ifsc}

ASSOCIATED SERVICES
----------------
Debit Card: {debit_card}
Net Banking: {net_banking}
Mobile Banking: {mobile_banking}
Recurring Instructions: {recurring_instructions}
"""

# PNB Account Closure Template
pnb_closure_template = """
**PUNJAB NATIONAL BANK - ACCOUNT CLOSURE FORM**
Reference Number: {ref_number}

ACCOUNT HOLDER DETAILS
-------------------
Name: {name}
Customer ID: {customer_id}
Mobile Number: {mobile}
Email: {email}

ACCOUNT INFORMATION
----------------
Account Number: {account_number}
Type of Account: {account_type}
Branch Name: {branch_name}
Opening Date: {opening_date}

CLOSURE REQUEST
-------------
Reason for Closure: {reason}
Settlement Method: {settlement_method}
Transfer Details (if applicable):
    Bank Name: {transfer_bank}
    Account Number: {transfer_account}
    IFSC Code: {transfer_ifsc}

LINKED FACILITIES
--------------
Debit Card Number: {debit_card}
Internet Banking: {internet_banking}
Mobile Banking: {mobile_banking}
Standing Instructions: {standing_instructions}
"""

# Yes Bank Account Closure Template
yes_bank_closure_template = """
**YES BANK - ACCOUNT CLOSURE REQUEST**
Form Number: {form_number}

CUSTOMER INFORMATION
-----------------
Customer Name: {customer_name}
Customer ID: {customer_id}
Contact Number: {contact_number}
Email ID: {email_id}

ACCOUNT DETAILS
-------------
Account Number: {account_number}
Account Type: {account_type}
Branch Location: {branch_location}
Account Opening Date: {opening_date}

CLOSURE INFORMATION
----------------
Primary Reason: {primary_reason}
Additional Comments: {comments}
Preferred Closure Date: {closure_date}
Settlement Mode: {settlement_mode}

LINKED PRODUCTS
-------------
Debit Card Details: {debit_card}
Digital Banking: {digital_banking}
Investment Products: {investments}
Locker Services: {locker_services}

SETTLEMENT DETAILS
---------------
Transfer Account Details:
    Bank Name: {transfer_bank}
    Account Number: {transfer_account}
    IFSC Code: {transfer_ifsc}
"""

# Canara Bank Account Closure Template
canara_closure_template = """
**CANARA BANK - ACCOUNT CLOSURE APPLICATION**
Application Number: {application_no}

ACCOUNT HOLDER DETAILS
-------------------
Name: {name}
Customer ID: {customer_id}
Mobile Number: {mobile}
Email ID: {email}

ACCOUNT INFORMATION
----------------
Account Number: {account_number}
Account Type: {account_type}
Branch: {branch}
IFSC Code: {ifsc}

CLOSURE DETAILS
-------------
Reason for Closure: {reason}
Requested Date: {requested_date}
Balance Settlement: {settlement_mode}
Transfer Details:
    Bank Name: {transfer_bank}
    Account Number: {transfer_account}
    IFSC Code: {transfer_ifsc}

LINKED SERVICES
-------------
Debit Card: {debit_card}
Net Banking: {net_banking}
Mobile Banking: {mobile_banking}
Standing Instructions: {standing_instructions}
"""

# Union Bank Account Closure Template
union_bank_closure_template = """
**UNION BANK OF INDIA - ACCOUNT CLOSURE FORM**
Reference Number: {ref_no}

CUSTOMER DETAILS
--------------
Full Name: {full_name}
Customer ID: {customer_id}
Contact Number: {contact_number}
Email Address: {email}

ACCOUNT DETAILS
-------------
Account Number: {account_number}
Account Type: {account_type}
Branch Name: {branch_name}
IFSC Code: {ifsc_code}

CLOSURE REQUEST
-------------
Reason for Closure: {reason}
Preferred Date: {preferred_date}
Mode of Settlement: {settlement_mode}
Settlement Account:
    Bank Name: {settlement_bank}
    Account Number: {settlement_account}
    IFSC Code: {settlement_ifsc}

LINKED FACILITIES
--------------
Debit Card Number: {debit_card}
Internet Banking: {internet_banking}
Mobile Banking: {mobile_banking}
Standing Instructions: {standing_instructions}
"""

# Bank of Baroda Account Closure Template
bob_closure_template = """
**BANK OF BARODA - ACCOUNT CLOSURE REQUEST**
Form Number: {form_number}

ACCOUNT HOLDER INFORMATION
-----------------------
Customer Name: {customer_name}
Customer ID: {customer_id}
Mobile Number: {mobile}
Email ID: {email}

ACCOUNT DETAILS
-------------
Account Number: {account_number}
Account Type: {account_type}
Home Branch: {home_branch}
IFSC Code: {ifsc}

CLOSURE INFORMATION
----------------
Reason for Closure: {reason}
Requested Closure Date: {closure_date}
Current Balance: {current_balance}
Settlement Method: {settlement_method}

SETTLEMENT DETAILS
---------------
Transfer Details:
    Bank Name: {transfer_bank}
    Account Number: {transfer_account}
    IFSC Code: {transfer_ifsc}
    Account Holder Name: {transfer_name}

LINKED SERVICES
-------------
Debit Card: {debit_card}
Internet Banking: {internet_banking}
Mobile Banking: {mobile_banking}
Standing Instructions: {standing_instructions}
""" 