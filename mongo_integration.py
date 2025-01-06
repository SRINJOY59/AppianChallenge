import warnings
import pymongo
from pymongo import MongoClient

class PushToMongo:
    def __init__(self):
        self.client = MongoClient("mongodb+srv://soumyadippaikgns:MJi3toN9QPhdpGPP@cluster0.nku1s.mongodb.net/")
        
        self.document_db = self.client["document_database"]
        self.user_info_db = self.client["User_info"]
        
        self.collections = {
            "financial": self.document_db["financial_data"],
            "bank": self.document_db["bank_data"],
            "identity": self.document_db["identity_data"],
            "receipt": self.document_db["receipt_data"]
        }

    def store_in_mongo(self, username, collection_type, relevant_info):

        
        document = {
            "relevant_information": relevant_info  
        }
        
        if collection_type not in self.collections:
            raise ValueError(f"Invalid collection type: {collection_type}. Must be one of {list(self.collections.keys())}")
        self.collections[collection_type].insert_one(document)
        
        user_collection = self.user_info_db[username]
        user_collection.insert_one(document)

warnings.filterwarnings("ignore")

# def main():
#     # Example input
#     test_data = [
#         {
#             "username": "alice123",
#             "email_id": "alice@example.com",
#             "collection_type": "Bank Data",
#             "relevant_information": {
#                 "account_number": "1234567890",
#                 "bank_name": "State Bank",
#                 "branch": "Downtown",
#                 "account_balance": "$2500"
#             }
#         },
#         {
#             "username": "bob456",
#             "email_id": "bob@example.com",
#             "collection_type": "Financial Data",
#             "relevant_information": {
#                 "investment_type": "Stocks",
#                 "portfolio_value": "$12000",
#                 "last_transaction": "2024-12-01"
#             }
#         },
#         {
#             "username": "alice123",  # Repeated username
#             "email_id": "alice@example.com",
#             "collection_type": "Receipt Data",
#             "relevant_information": {
#                 "transaction_id": "TXN123456789",
#                 "amount": "$50",
#                 "vendor": "CoffeeShop",
#                 "date": "2024-12-20"
#             }
#         },
#         {
#             "username": "carol789",
#             "email_id": "carol@example.com",
#             "collection_type": "Identity Data",
#             "relevant_information": {
#                 "id_type": "Driver's License",
#                 "id_number": "DL98765432",
#                 "issue_date": "2018-06-15",
#                 "expiry_date": "2028-06-14"
#             }
#         },
#         {
#             "username": "bob456",  # Repeated username
#             "email_id": "bob@example.com",
#             "collection_type": "Bank Data",
#             "relevant_information": {
#                 "account_number": "5432167890",
#                 "bank_name": "Global Bank",
#                 "branch": "City Center",
#                 "account_balance": "$7500",
#                 "last_transaction": "2024-12-23"
#             }
#         }
#     ]

    
#     push_to_mongo = PushToMongo()
#     for json_data in test_data:
#         push_to_mongo.store_in_mongo(json_data)
#         print(f"Data for {json_data['username']} inserted successfully.")


# if __name__ == "__main__":
#     main()
