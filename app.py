import os
import re
import json
import logging
from datetime import datetime
import dateparser
from flask import Flask, request, jsonify
from word2number import w2n

# Import the official Groq client
from groq import Groq

# Import SentenceTransformer for vector embeddings
from sentence_transformers import SentenceTransformer, util

# Import PyMongo and ObjectId for MongoDB integration
from pymongo import MongoClient
from bson.tz_util import utc

from twilio.twiml.messaging_response import MessagingResponse

# -------------------------------
# Hardcoded Environment Variables
# -------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# MONGO_URI = "mongodb://localhost:27017"
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB_NAME = os.environ.get("expense_tracker1")
MONGO_COLLECTION = os.environ.get("expenses")

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------
# MongoDB Client Class using PyMongo (Local Connection)
# -------------------------------
class MongoDBClient:
    def __init__(self, uri, db_name, collection_name):
        logging.info(f"Connecting to MongoDB at {uri}...")
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, directConnection=False)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.client.admin.command("ping")
            logging.info("✅ Successfully connected to MongoDB!")
        except Exception as e:
            logging.error(f"❌ MongoDB Connection Failed: {e}")
            raise e

    def insert(self, record):
        result = self.collection.insert_one(record)
        record["expense_id"] = str(result.inserted_id)
        return record["expense_id"]

    def find(self, query):
        results = list(self.collection.find(query))
        for rec in results:
            rec["expense_id"] = str(rec.get("_id"))
        return results

    def delete(self, query):
        expense_id = query.get("expense_id")
        if expense_id:
            result = self.collection.delete_one({"_id": ObjectId(expense_id), "user_id": query.get("user_id")})
            return result.deleted_count > 0
        return False

db_client = MongoDBClient(MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION)

# -------------------------------
# Load Sentence Transformer Model and Precompute Canonical Category Embeddings
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
canonical_categories = [
    "ATM", "Auto", "Bars", "Beauty", "Cashback", "Clothing", "Coffee Shops",
    "Credit Card Payment", "Education", "Entertainment", "Fees", "Food",
    "Fuel", "Gifts", "Groceries", "Gym", "Home", "Housing", "Income",
    "Insurance", "Medical", "Pets", "Pharmacy", "Restaurants", "Services",
    "Shopping", "Streaming", "Taxes", "Technology", "Transportation",
    "Travel", "Transfer", "Utilities", "Beverages", "Other"
]
category_embeddings = embedding_model.encode(canonical_categories, convert_to_tensor=True)

def get_category_vector(item, threshold=0.5):
    item_embedding = embedding_model.encode(item, convert_to_tensor=True)
    cosine_scores = util.cos_sim(item_embedding, category_embeddings)
    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[0][best_idx].item()
    if best_score < threshold:
        logging.info(f"Low confidence for '{item}' (score={best_score:.2f}); falling back to LLM mapping.")
        return None
    return canonical_categories[best_idx]

# -------------------------------
# Legacy Category Mapping (Fallback)
# -------------------------------
CATEGORY_MAPPING = {
    "ATM": ["atm", "cash", "withdraw"],
    "Auto": ["auto body"],
    "Bars": ["bar", "pubs", "irish", "brewery"],
    "Beauty": ["body"],
    "Cashback": ["cashback", "reward", "bonus", "cash back"],
    "Clothing": ["clothing", "shoes", "accessories", "tshirt", "t-shirts"],
    "Coffee Shops": ["coffee", "cafe", "tea", "starbucks", "dunkin"],
    "Credit Card Payment": ["card payment", "autopay"],
    "Education": ["kindle", "tuition"],
    "Entertainment": ["event", "show", "movies", "cinema", "theater"],
    "Fees": ["fee"],
    "Food": ["snack", "donalds", "burger king", "kfc", "subway", "pizza", "domino", "taco bell", "wendy", "chick-fil-a", "popeyes", "arby's", "chipotle"],
    "Fuel": ["fuel", "gas", "petrol"],
    "Gifts": ["donation", "gift"],
    "Groceries": ["groceries", "supermarket", "food", "familia"],
    "Gym": ["gym", "fitness", "yoga", "pilates", "crossfit"],
    "Home": ["ikea"],
    "Housing": ["rent", "mortgage"],
    "Income": ["refund", "deposit", "paycheck"],
    "Insurance": ["insurance"],
    "Medical": ["medical", "doctor", "dentist", "hospital", "clinic"],
    "Pets": ["vet", "veterinary", "pet", "dog", "cat"],
    "Pharmacy": ["pharmacy", "drugstore", "cvs", "walgreens", "rite aid", "duane"],
    "Restaurants": ["restaurant", "lunch", "dinner"],
    "Services": ["service", "laundry", "dry cleaning"],
    "Shopping": ["shopping", "amazon", "walmart", "target", "safeway"],
    "Streaming": ["netflix", "spotify", "hulu", "hbo"],
    "Taxes": ["tax", "irs"],
    "Technology": ["technology", "software", "hardware", "electronics"],
    "Transportation": ["bus", "train", "subway", "metro", "airline", "uber", "lyft", "taxi", "ola", "cab"],
    "Travel": ["travel", "holiday", "trip", "airbnb", "kiwi", "hotel", "hostel", "resort", "kayak", "expedia", "booking.com"],
    "Transfer": ["payment from"],
    "Utilities": ["electricity", "water", "gas", "ting", "verizon", "comcast", "sprint", "t-mobile", "at&t", "mint"],
    "Beverages": ["beverages", "coldrink", "soda", "juice", "water", "beer", "wine"],
    "Other": []
}

def get_category(item):
    canonical_item = item.lower()
    for category, keywords in CATEGORY_MAPPING.items():
        for keyword in keywords:
            if keyword in canonical_item:
                return category
    return "Other"

# -------------------------------
# Helper Function: Convert Number Words to Digits
# -------------------------------
def convert_number_words(text):
    number_word_list = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion"
    ]
    pattern = r"\b(?:" + "|".join(number_word_list) + r")(?:[-\s]+(?:" + "|".join(number_word_list) + r"))*\b"
    
    def replace_match(match):
        phrase = match.group(0)
        try:
            number = w2n.word_to_num(phrase)
            return str(number)
        except Exception as e:
            logging.error(f"Error converting phrase '{phrase}': {e}")
            return phrase

    converted_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    logging.debug(f"Converted text: {converted_text}")
    return converted_text

# -------------------------------
# GroqLLMClient Class
# -------------------------------
class GroqLLMClient:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
    
    def get_response(self, prompt, model="llama-3.3-70b-versatile"):
        logging.debug(f"Groq LLM Prompt: {prompt}")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model
            )
            text = chat_completion.choices[0].message.content.strip()
            logging.debug(f"Groq LLM Response: {text}")
            return text
        except Exception as e:
            logging.error(f"Groq LLM API error: {e}")
            raise e
    
    def transcribe_voice(self, media_url):
        logging.debug(f"Transcribing voice from URL: {media_url}")
        try:
            transcription = self.client.audio.transcriptions.create(
                media_url=media_url,
                model="whisper-large-v3-turbo",
                prompt="Transcribe the following voice note.",
                response_format="json",
                language="en",
                temperature=0.0
            )
            text = transcription
            logging.debug(f"Voice transcription result: {text}")
            return text
        except Exception as e:
            logging.error(f"Error transcribing voice: {e}")
            raise e


llm_client = GroqLLMClient(GROQ_API_KEY)

# -------------------------------
# ExpenseTracker Class
# -------------------------------
class ExpenseTracker:
    def __init__(self, llm_client, db_client):
        self.llm_client = llm_client
        self.db = db_client

    def correct_typos(self, message):
        prompt = (
            "Please carefully review the following message for any spelling, grammatical, or typographical errors. "
            "Correct all errors while preserving the exact original meaning and tone. Do not alter the structure or content "
            "of the message, and return only the corrected text without any additional commentary. "
            "For example, if you see 'spenrt', correct it to 'spent'.\n"
            f"Message: \"{message}\""
        )
        try:
            corrected = self.llm_client.get_response(prompt)
            if corrected:
                logging.info(f"Typo correction: '{message}' corrected to '{corrected}'")
                return corrected
        except Exception as e:
            logging.error(f"Typo correction failed: {e}")
        return message

    def hybrid_map_category(self, word):
        word = word.lower().strip()
        logging.debug(f"Mapping '{word}' dynamically using hybrid approach.")
        try:
            vector_category = get_category_vector(word, threshold=0.5)
            if vector_category is not None:
                logging.info(f"Vector similarity mapping: {word} -> {vector_category}")
                return vector_category
            else:
                raise ValueError("Cosine similarity below threshold.")
        except Exception as e:
            logging.error(f"Vector similarity mapping failed for {word}: {e}")
            prompt = (
                f"Map the following expense item to one of the canonical categories: {', '.join(canonical_categories)}. "
                "Return only the category in lowercase.\n"
                f"Input: \"{word}\""
            )
            try:
                response = self.llm_client.get_response(prompt)
                canonical = response.strip().lower()
                logging.info(f"LLM mapping fallback: {word} -> {canonical}")
                return canonical
            except Exception as e:
                logging.error(f"LLM mapping failed for {word}: {e}")
                return word

    def process_message_with_llm(self, message, mode):
        current_date = datetime.now().strftime("%Y-%m-%d")
        if mode == "add":
            prompt = (
                "Extract all expense entries from the following message. Return a JSON list where each expense "
                "contains the keys 'amount', 'item', and 'date' (if mentioned) and any relevant 'context' (e.g., phrases like 'for the party'). "
                "When you encounter relative dates such as 'yesterday', 'tomorrow', 'day after tomorrow', or 'day before yesterday', "
                "convert them to absolute dates using the following reference: assume today's date is " + current_date + ". "
                "Return only the JSON output without any extra commentary.\n\n"
                f"Message: \"{message}\""
            )
        elif mode == "query":
            prompt = (
                "Extract query details from the following message. Return a JSON object with keys 'items' (a list of items/categories being queried) "
                "and, if a date or month is mentioned, a key 'time_range'. Use the reference date " + current_date + " for any relative date interpretation. "
                "Return only the JSON output without any extra commentary.\n\n"
                f"Message: \"{message}\""
            )
        elif mode == "delete":
            prompt = (
                "Extract the expense ID to delete from the following message. Return a JSON object with the key 'expense_id'. "
                "Return only the JSON output without any extra commentary.\n\n"
                f"Message: \"{message}\""
            )
        else:
            return {}
        try:
            response = self.llm_client.get_response(prompt)
            data = json.loads(response)
            logging.debug(f"LLM response parsed: {data}")
            return data
        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return self.regex_fallback(message, mode)

    def regex_fallback(self, message, mode):
        logging.debug(f"Using regex fallback for mode {mode}: {message}")
        if mode == "query":
            items_match = re.search(r"(?:for|on)\s+(.+)", message, re.IGNORECASE)
            if items_match:
                items_str = items_match.group(1).strip()
                items_str = re.sub(r'[^\w\s]', '', items_str)
                items = [item.strip().lower() for item in items_str.split("and")]
                fallback_query = {"items": items, "time_range": None}
            else:
                fallback_query = {"items": ["all"], "time_range": None}
            logging.debug(f"Regex fallback query: {fallback_query}")
            return fallback_query
        elif mode == "add":
            processed_message = convert_number_words(message)
            pattern = r"(?:(\d+)\s*(?:rs|\$|₹|dollars|rupees)?\s*(?:on|for)\s*([\w\s]+))|(\d+)\s*(?:rs|\$|₹|dollars|rupees)?\b"
            matches = re.findall(pattern, processed_message, flags=re.IGNORECASE)
            entries = []
            for match in matches:
                if match[0]:
                    amount = match[0]
                    item = match[1].strip()
                elif match[2]:
                    amount = match[2]
                    item = "Other"
                else:
                    continue
                date_match = re.search(r"\b(?:on|for)\s+([\w\-\: ]+)", processed_message)
                if date_match:
                    date_str = date_match.group(1)
                    parsed_date = dateparser.parse(date_str)
                    date_val = parsed_date.strftime("%Y-%m-%d") if parsed_date else datetime.now().strftime("%Y-%m-%d")
                else:
                    date_val = datetime.now().strftime("%Y-%m-%d")
                entries.append({"amount": float(amount), "item": item, "date": date_val})
            logging.debug(f"Regex fallback add entries: {entries}")
            return entries
        elif mode == "delete":
            pattern = r"delete(?: expense)? (\d+)"
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                fallback_delete = {"expense_id": int(match.group(1))}
                logging.debug(f"Regex fallback delete: {fallback_delete}")
                return fallback_delete
            return {}

    def add_expense(self, message, user_id):
        corrected_message = self.correct_typos(message)
        expenses = self.process_message_with_llm(corrected_message, mode="add")
        if not expenses:
            return "No valid expense entries found."
        responses = []
        for entry in expenses:
            if entry.get("amount") is None:
                return "Missing amount in the expense entry. Please include the amount."
            raw_amount = str(entry["amount"])
            clean_amount = re.sub(r'[^\d\.]', '', raw_amount)
            try:
                amount_value = float(clean_amount)
            except ValueError:
                return "Invalid amount format. Please ensure the amount is numeric."
            if amount_value < 0:
                return "Negative amounts are not allowed. Please provide a valid positive amount."
            date_str = entry.get("date", "")
            parsed_date = dateparser.parse(date_str) if date_str else None
            absolute_date = parsed_date.strftime("%Y-%m-%d") if parsed_date else datetime.now().strftime("%Y-%m-%d")
            original_item = entry["item"].strip()
            canonical = self.hybrid_map_category(original_item)
            mapped_category = get_category(canonical)
            expense_record = {
                "user_id": user_id,
                "amount": amount_value,
                "item": original_item,
                "category": mapped_category,
                "context": entry.get("context", "").split() if entry.get("context") else [],
                "date": absolute_date,
                "timestamp": datetime.now().timestamp()
            }
            expense_id = self.db.insert(expense_record)
            responses.append(f"Expense added: Rs {amount_value} for {original_item} on {absolute_date} (ID: {expense_id}).")
        return "\n".join(responses)

    def query_expense(self, message, user_id):
        corrected_message = self.correct_typos(message)
        query_data = self.process_message_with_llm(corrected_message, mode="query")
        if "items" not in query_data or not query_data["items"]:
            query_data = {"items": ["all"], "time_range": None}
        total = 0
        response_details = []
        for item in query_data.get("items", []):
            query_term = item.strip().lower()
            if query_term in ["all", "expenses", "expense"]:
                filtered_records = self.db.find({"user_id": user_id})
            else:
                all_records = self.db.find({"user_id": user_id})
                exact_matches = [rec for rec in all_records if rec["item"].strip().lower() == query_term]
                if exact_matches:
                    filtered_records = exact_matches
                else:
                    canonical = self.hybrid_map_category(item)
                    category = get_category(canonical)
                    filtered_records = self.db.find({"user_id": user_id, "category": category})
                if query_data.get("time_range") and query_term not in ["all", "expenses", "expense"]:
                    try:
                        reference_date = datetime.now()
                        absolute_date = dateparser.parse(query_data["time_range"], settings={'RELATIVE_BASE': reference_date})
                        if absolute_date:
                            filtered_records = [rec for rec in filtered_records if rec["date"] == absolute_date.strftime("%Y-%m-%d")]
                    except Exception as e:
                        logging.error(f"Error parsing relative date '{query_data['time_range']}': {e}")
            for rec in filtered_records:
                response_details.append(
                    f"ID: {rec['expense_id']}, Amount: Rs {rec['amount']}, Item: {rec['item']}, "
                    f"Category: {rec['category']}, Date: {rec['date']}"
                )
            total += sum(rec["amount"] for rec in filtered_records)
        response_details.append(f"Total: Rs {total}")
        return "\n".join(response_details)

    def delete_expense(self, message, user_id):
        corrected_message = self.correct_typos(message)
        query_data = self.process_message_with_llm(corrected_message, mode="delete")
        expense_id = query_data.get("expense_id")
        if not expense_id:
            return "No expense ID provided for deletion."
        result = self.db.delete({"user_id": user_id, "expense_id": expense_id})
        if result:
            return f"Expense {expense_id} deleted."
        return f"Expense {expense_id} not found."

    def classify_intent(self, message):
        prompt = (
            "Classify the intent of the following message as one of: 'add expense', 'query expense', or 'delete expense'. "
            "Return only the intent as a single word: add, query, or delete.\n"
            f"Message: \"{message}\""
        )
        try:
            intent = self.llm_client.get_response(prompt).strip().lower()
            logging.debug(f"LLM intent classification: {intent}")
            if 'add' in intent or 'lend' in intent:
                return "add"
            elif 'query' in intent or 'show' in intent or 'list' in intent or 'how much' in intent:
                return "query"
            elif 'delete' in intent:
                return "delete"
            else:
                return "unknown"
        except Exception as e:
            logging.error(f"Intent classification failed: {e}")
            if re.search(r"(show|list|how much)", message, re.IGNORECASE):
                return "query"
            elif re.search(r"(spent|bought|add|lended)", message, re.IGNORECASE):
                return "add"
            elif re.search(r"delete", message, re.IGNORECASE):
                return "delete"
            return "unknown"

    def process_command(self, message, user_id):
        corrected_message = self.correct_typos(message)
        logging.info(f"Processing command: {corrected_message} for user: {user_id}")
        intent = self.classify_intent(corrected_message)
        logging.info(f"Detected intent: {intent}")
        if intent == "query":
            return self.query_expense(corrected_message, user_id)
        elif intent == "add":
            return self.add_expense(corrected_message, user_id)
        elif intent == "delete":
            return self.delete_expense(corrected_message, user_id)
        else:
            return "I didn’t understand that. Try 'add expense' or 'show expenses.'"

llm_client = GroqLLMClient(GROQ_API_KEY)
db_client = MongoDBClient(MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION)
tracker = ExpenseTracker(llm_client, db_client)

app = Flask(__name__)

@app.route('/process_command', methods=['POST'])
def process_command_route():
    data = request.get_json()
    message = data.get("message", "")
    user_id = data.get("user_id", "default_user")
    logging.info(f"Received command: {message} for user: {user_id}")
    response = tracker.process_command(message, user_id)
    return jsonify({"response": response})

@app.route('/whatsapp', methods=['POST'])
def whatsapp_reply():
    incoming_msg = request.form.get('Body', '').strip()
    from_number = request.form.get('From', '')
    user_id = from_number
    logging.info(f"Received WhatsApp message from {user_id}: {incoming_msg}")
    media_url = request.form.get('MediaUrl0', '')
    media_type = request.form.get('MediaContentType0', '')
    # If an audio file is provided, download and transcribe it.
    if media_url and "audio" in media_type.lower():
        try:
            incoming_msg = llm_client.transcribe_voice(media_url)
            logging.info(f"Transcribed voice message: {incoming_msg}")
        except Exception as e:
            logging.error(f"Voice transcription failed: {e}")
            incoming_msg = "Voice transcription failed."
    response_text = tracker.process_command(incoming_msg, user_id)
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response_text)
    return str(resp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
