# SBI Debit Card Application Template
sbi_debit_card_template = """
**STATE BANK OF INDIA - DEBIT CARD APPLICATION**
Application Reference: {ref_number}
Branch Code: {branch_code}

ACCOUNT DETAILS
--------------
Account Holder Name: {name}
Account Number: {account_number}
Account Type: {account_type}  # Savings/Current/NRE/NRO
Customer ID: {customer_id}
Branch Name: {branch_name}

PERSONAL DETAILS
--------------
Date of Birth: {dob}
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {mobile}
Email ID: {email}

CARD REQUIREMENTS
---------------
Card Variant: {card_variant}  # Global/Classic/Gold/Platinum
Card Network: {card_network}  # RuPay/Visa/Mastercard
International Usage: {international_enabled}  # Yes/No
Daily ATM Limit: {atm_limit}
Daily POS Limit: {pos_limit}
"""

# HDFC Debit Card Application Template
hdfc_debit_card_template = """
**HDFC BANK - DEBIT CARD APPLICATION**
Application ID: {application_id}

ACCOUNT INFORMATION
-----------------
Account Holder: {account_holder}
Account Number: {account_number}
Account Type: {account_type}
Customer ID: {customer_id}
Home Branch: {home_branch}

PERSONAL DETAILS
--------------
Date of Birth: {dob}
PAN Number: {pan}
Mobile Number: {mobile}
Email Address: {email}

CARD PREFERENCES
--------------
Card Type: {card_type}  # EasyShop/Premium/Business
Card Design: {card_design}  # Standard/Custom
ATM Withdrawal Limit: {withdrawal_limit}
Shopping Limit: {shopping_limit}
SMS Alerts: {sms_alerts}  # Yes/No
"""

# ICICI Debit Card Application Template
icici_debit_card_template = """
**ICICI BANK - DEBIT CARD APPLICATION**
Reference Number: {ref_no}

ACCOUNT DETAILS
-------------
Primary Account Holder: {primary_holder}
Account Number: {account_number}
Account Type: {account_type}
Customer ID: {customer_id}
Branch: {branch}

PERSONAL INFORMATION
------------------
Date of Birth: {dob}
PAN Card Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {mobile}
Email ID: {email}

CARD REQUIREMENTS
---------------
Card Variant: {card_variant}  # Coral/Sapphiro/Expression
Card Network: {network}  # Visa/Mastercard/RuPay
International Transactions: {international_enabled}
Daily Transaction Limits:
    ATM: {atm_limit}
    POS: {pos_limit}
    E-commerce: {ecom_limit}
"""

# Axis Bank Debit Card Application Template
axis_debit_card_template = """
**AXIS BANK - DEBIT CARD APPLICATION**
Application Number: {app_number}

ACCOUNT INFORMATION
-----------------
Account Name: {account_name}
Account Number: {account_number}
Account Type: {account_type}
CIF Number: {cif_number}
Branch: {branch}

CONTACT DETAILS
-------------
Mobile Number: {mobile}
Email ID: {email}
Communication Address: {address}

CARD PREFERENCES
--------------
Card Variant: {card_variant}  # Easy/Prime/Priority
Card Network: {network}
Daily Limits:
    Cash Withdrawal: {cash_limit}
    Purchase: {purchase_limit}
E-commerce Enabled: {ecom_enabled}  # Yes/No
"""

# Kotak Mahindra Bank Debit Card Application Template
kotak_debit_card_template = """
**KOTAK MAHINDRA BANK - DEBIT CARD APPLICATION**
Application ID: {application_id}

ACCOUNT DETAILS
-------------
Account Holder Name: {holder_name}
Account Number: {account_number}
Customer Relationship Number: {crn}
Branch: {branch}

PERSONAL INFORMATION
------------------
Date of Birth: {dob}
PAN Number: {pan}
Mobile Number: {mobile}
Email Address: {email}

DEBIT CARD REQUIREMENTS
--------------------
Card Type: {card_type}  # Essential/Pro/League
Network Choice: {network}
Daily Limits Required:
    ATM: {atm_limit}
    POS: {pos_limit}
International Usage: {intl_usage}  # Yes/No
"""

# Punjab National Bank Debit Card Application Template
pnb_debit_card_template = """
**PUNJAB NATIONAL BANK - DEBIT CARD APPLICATION**
Application Number: {app_number}

ACCOUNT DETAILS
-------------
Name: {name}
Account Number: {account_number}
Customer ID: {customer_id}
Branch: {branch}
IFSC Code: {ifsc}

PERSONAL DETAILS
--------------
Date of Birth: {dob}
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {mobile}

CARD REQUIREMENTS
---------------
Card Type: {card_type}  # RuPay Classic/Platinum/Premium
ATM Withdrawal Limit: {withdrawal_limit}
POS Transaction Limit: {pos_limit}
E-commerce Enabled: {ecom_enabled}
"""

# Yes Bank Debit Card Application Template
yes_bank_debit_card_template = """
**YES BANK - DEBIT CARD APPLICATION**
Reference ID: {ref_id}

ACCOUNT INFORMATION
-----------------
Account Name: {account_name}
Account Number: {account_number}
Customer ID: {customer_id}
Branch Name: {branch_name}

KYC DETAILS
----------
Date of Birth: {dob}
PAN Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {mobile}
Email: {email}

CARD SELECTION
------------
Card Variant: {card_variant}  # Classic/Gold/Platinum
Network Preference: {network}
Daily Limits:
    ATM Withdrawal: {atm_limit}
    Shopping: {shopping_limit}
International Enabled: {intl_enabled}
"""

# Canara Bank Debit Card Application Template
canara_debit_card_template = """
**CANARA BANK - DEBIT CARD APPLICATION**
Application Number: {application_no}

ACCOUNT DETAILS
-------------
Account Holder Name: {holder_name}
Account Number: {account_number}
Customer ID: {customer_id}
Branch: {branch}
IFSC: {ifsc}

PERSONAL DETAILS
--------------
Date of Birth: {dob}
PAN Number: {pan}
Mobile Number: {mobile}
Email ID: {email}

CARD REQUIREMENTS
---------------
Card Type: {card_type}  # Classic/Gold/Platinum
Network: {network}  # RuPay/Visa/Mastercard
Daily Withdrawal Limit: {withdrawal_limit}
POS/Online Limit: {pos_limit}
"""

# Union Bank Debit Card Application Template
union_bank_debit_card_template = """
**UNION BANK OF INDIA - DEBIT CARD APPLICATION**
Application Reference: {app_ref}

ACCOUNT INFORMATION
-----------------
Account Holder: {account_holder}
Account Number: {account_number}
CIF Number: {cif_number}
Branch: {branch}

PERSONAL DETAILS
--------------
Date of Birth: {dob}
PAN Number: {pan}
Mobile Number: {mobile}
Email ID: {email}

CARD PREFERENCES
--------------
Card Variant: {card_variant}  # Classic/Gold/Platinum
Card Network: {network}
Transaction Limits:
    ATM: {atm_limit}
    POS/E-com: {pos_limit}
International Usage: {intl_usage}
"""

# Bank of Baroda Debit Card Application Template
bob_debit_card_template = """
**BANK OF BARODA - DEBIT CARD APPLICATION**
Reference Number: {ref_number}

ACCOUNT DETAILS
-------------
Account Name: {account_name}
Account Number: {account_number}
Customer ID: {customer_id}
Branch Name: {branch_name}

PERSONAL INFORMATION
------------------
Date of Birth: {dob}
PAN Card Number: {pan}
Aadhaar Number: {aadhaar}
Mobile Number: {mobile}
Email: {email}

DEBIT CARD REQUIREMENTS
--------------------
Card Type: {card_type}  # Classic/Gold/Platinum
Network Choice: {network}
Daily Limits:
    ATM Withdrawal: {atm_limit}
    POS/E-commerce: {pos_limit}
SMS Alert Service: {sms_alert}  # Yes/No
International Usage: {intl_enabled}  # Yes/No
"""
