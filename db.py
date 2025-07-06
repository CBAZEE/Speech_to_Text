import pandas as pd
from pymongo import MongoClient

# === Step 1: Read CSV ===
df = pd.read_csv("jay_mataji.csv")  # replace with your file name

# === Step 2: Connect to MongoDB Atlas ===
client = MongoClient(
    "mongodb+srv://Jay_Dobariya:jaypatel9959@cluster0.atpwjkz.mongodb.net/")
db = client["database"]  # database name
collection = db["my_collection"]  # collection name

# === Step 3: Insert data ===
data = df.to_dict(orient="records")
collection.insert_many(data)

print("✅ Data uploaded successfully.")
