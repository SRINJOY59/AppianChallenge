# SBI Passbook Template
sbi_passbook_template = """
**STATE BANK OF INDIA - PASSBOOK**
Branch: {branch_name}
IFSC: {ifsc_code}

ACCOUNT DETAILS
--------------
Account Holder: {account_holder}
Account Number: {account_number}
Account Type: {account_type}  # Savings/Current/NRI
Customer ID: {customer_id}
Opening Date: {opening_date}

TRANSACTION FORMAT
----------------
Date | Value Date | Description | Cheque No. | Debit | Credit | Balance
{date} | {value_date} | {description} | {cheque_no} | {debit} | {credit} | {balance}

Note: This is a computer generated statement
"""

# HDFC Bank Passbook Template
hdfc_passbook_template = """
**HDFC BANK - PASSBOOK**
Branch: {branch_name}
IFSC Code: {ifsc_code}

ACCOUNT INFORMATION
-----------------
Name: {account_holder}
Account No.: {account_number}
Product Type: {account_type}  # Savings/Current/Salary
Mode of Operation: {operation_mode}  # Single/Joint
Mobile: {mobile_number}

TRANSACTION DETAILS
-----------------
Posting Date | Transaction Details | Ref No./Cheque | Withdrawal | Deposit | Balance
{date} | {details} | {ref_no} | {withdrawal} | {deposit} | {balance}

This is an electronic statement of your account
"""

# ICICI Bank Passbook Template
icici_passbook_template = """
**ICICI BANK - PASSBOOK**
Branch: {branch_name}
IFSC: {ifsc_code}

ACCOUNT HOLDER DETAILS
--------------------
Customer Name: {customer_name}
Account Number: {account_number}
Account Type: {account_type}
Customer ID: {customer_id}
CKYC Number: {ckyc_number}

TRANSACTION RECORD
----------------
Date | Particulars | Ref No. | Withdrawals | Deposits | Closing Balance
{date} | {particulars} | {ref_no} | {withdrawals} | {deposits} | {closing_balance}

For any queries, please contact your branch
"""

# Axis Bank Passbook Template
axis_passbook_template = """
**AXIS BANK - PASSBOOK**
Branch Name: {branch_name}
IFSC Code: {ifsc_code}

ACCOUNT DETAILS
-------------
Account Name: {account_name}
Account No: {account_number}
Product: {product_type}  # Savings/Current/Salary
Customer ID: {customer_id}
Mobile No.: {mobile_number}

TRANSACTION ENTRIES
-----------------
Tran Date | Description | Ref Number | Withdrawal | Deposit | Balance
{date} | {description} | {ref_number} | {withdrawal} | {deposit} | {balance}

This is a system generated statement
"""

# Kotak Mahindra Bank Passbook Template
kotak_passbook_template = """
**KOTAK MAHINDRA BANK - PASSBOOK**
Home Branch: {home_branch}
IFSC: {ifsc_code}

ACCOUNT INFORMATION
-----------------
Account Holder: {account_holder}
Account Number: {account_number}
Scheme Code: {scheme_code}
Customer Relationship No: {crn}
Account Opening Date: {opening_date}

TRANSACTION DETAILS
-----------------
Post Date | Value Date | Transaction Description | Chq/Ref No | Debit | Credit | Balance
{post_date} | {value_date} | {description} | {ref_no} | {debit} | {credit} | {balance}

Please inform the bank of any change in your contact details
"""

# Punjab National Bank Passbook Template
pnb_passbook_template = """
**PUNJAB NATIONAL BANK - PASSBOOK**
Branch: {branch_name}
Branch Code: {branch_code}
IFSC: {ifsc_code}

ACCOUNT DETAILS
-------------
Name: {account_holder}
Account No.: {account_number}
Type: {account_type}
Customer ID: {customer_id}
Mobile: {mobile_number}

TRANSACTION RECORD
---------------
Date | Particulars | Instrument No | Debit | Credit | Balance
{date} | {particulars} | {instrument_no} | {debit} | {credit} | {balance}

Keep your passbook updated regularly
"""

# Yes Bank Passbook Template
yes_bank_passbook_template = """
**YES BANK - PASSBOOK**
Branch: {branch_name}
IFSC Code: {ifsc_code}

ACCOUNT HOLDER INFORMATION
-----------------------
Account Name: {account_name}
Account Number: {account_number}
Product Type: {product_type}
Customer ID: {customer_id}
Statement Period: {statement_period}

TRANSACTION DETAILS
----------------
Transaction Date | Narration | Reference | Withdrawals | Deposits | Balance
{date} | {narration} | {reference} | {withdrawals} | {deposits} | {balance}

For any assistance, please call YES BANK 24/7 at 1800 1200
"""

# Canara Bank Passbook Template
canara_passbook_template = """
**CANARA BANK - PASSBOOK**
Branch: {branch_name}
IFSC Code: {ifsc_code}

ACCOUNT DETAILS
-------------
Name of Account Holder: {account_holder}
Account Number: {account_number}
Type of Account: {account_type}
Customer ID: {customer_id}
Mobile Number: {mobile_number}

TRANSACTION ENTRIES
----------------
Date | Description | Ref.No/Cheque | Debit | Credit | Balance
{date} | {description} | {ref_no} | {debit} | {credit} | {balance}

Please keep your passbook updated
"""

# Union Bank Passbook Template
union_bank_passbook_template = """
**UNION BANK OF INDIA - PASSBOOK**
Branch Name: {branch_name}
IFSC: {ifsc_code}

ACCOUNT INFORMATION
-----------------
Account Holder Name: {holder_name}
Account Number: {account_number}
Account Type: {account_type}
CIF Number: {cif_number}
Mobile: {mobile_number}

TRANSACTION DETAILS
----------------
Date | Transaction Description | Cheque/Ref No. | Withdrawal | Deposit | Balance
{date} | {description} | {ref_no} | {withdrawal} | {deposit} | {balance}

Keep your mobile number updated for SMS alerts
"""

# Bank of Baroda Passbook Template
bob_passbook_template = """
**BANK OF BARODA - PASSBOOK**
Branch: {branch_name}
IFSC Code: {ifsc_code}

ACCOUNT DETAILS
-------------
Customer Name: {customer_name}
Account Number: {account_number}
Scheme Code: {scheme_code}
Customer ID: {customer_id}
Account Opening Date: {opening_date}

TRANSACTION RECORD
---------------
Tran Date | Particulars | Instrument No | Debit | Credit | Balance
{tran_date} | {particulars} | {instrument_no} | {debit} | {credit} | {balance}

Please quote your account number in all correspondence
"""
