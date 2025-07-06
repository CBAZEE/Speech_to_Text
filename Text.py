import csv
import json
import sounddevice as sd
import queue
import vosk
from sentence_transformers import SentenceTransformer, util
import re
import os

# Load MiniLM model
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# File paths
csv_file = 'jay_mataji.csv'
json_file = 'jay_mataji.json'
json_path = "D:/Speech to text/jay_mataji.json"

# Convert CSV to JSON
with open(csv_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data_csv = list(reader)

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data_csv, f, ensure_ascii=False, indent=4)

print(f"✅ Converted and saved to {json_file}")

# Load JSON
if not os.path.exists(json_path):
    print(f"❌ File not found: {json_path}")
    exit()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load Vosk model
model = vosk.Model("D:/Speech to text/vosk-model-small-en-us-0.15")
q = queue.Queue()

# Fields to include in output
financial_fields = [
    "no.", "company", "date", "purchase",
    "selling", "total income", "total expense",
    "profit", "loss"
]
FIELDS = financial_fields


def callback(indata, frames, time, status):
    q.put(bytes(indata))


def recognize_speech():
    print("🎤 Press Enter to use MIC or type your query directly...")
    typed_input = input("⌨️ Type your query (or press Enter to speak): ").strip()

    if typed_input:
        if "shop" in typed_input.lower():
            print("🛍️ Shop command detected. Opening shop...")
            os.system("start https://www.google.com/search?q=tyre+shop")
            return None
        return typed_input.lower()

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, 16000)
        print("🎧 Listening... (say 'stop' to exit)")
        while True:
            data_audio = q.get()
            if rec.AcceptWaveform(data_audio):
                result_json = rec.Result()
                result = json.loads(result_json)
                text = result.get("text", "")
                print(f"📝 You said: {text}")

                if "stop" in text.lower():
                    print("🛑 Stop command detected. Exiting...")
                    return None

                if "shop" in text.lower():
                    print("🛍️ Shop command detected. Opening shop...")
                    os.system("start https://www.google.com/search?q=tyre+shop")
                    continue

                return text.lower()


def get_numeric_value(entry, field):
    value_str = entry.get(field, '0')
    try:
        return float(value_str)
    except:
        return 0.0


def match_field_with_minilm(query):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    field_embs = embed_model.encode(financial_fields, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, field_embs)[0]
    best_match_idx = scores.argmax().item()
    return financial_fields[best_match_idx]


def search_company_by_name(text):
    results = []
    for entry in data:
        company = entry.get("company", "").lower()
        if company and company in text.lower():
            results.append(entry)
        elif text.lower() in company:
            results.append(entry)
    return results


def filter_data_by_condition(field, operator, value):
    results = []
    for entry in data:
        entry_value = get_numeric_value(entry, field)
        if operator == "above" and entry_value > value:
            results.append(entry)
        elif operator == "below" and entry_value < value:
            results.append(entry)
        elif operator == "equal" and entry_value == value:
            results.append(entry)
    return results


def parse_condition_query(text):
    # Detect 'profit above 100', etc.
    patterns = [
        (r"(profit|loss|selling|purchase|total income|total expense)\s+(above|greater than|more than)\s+(\d+)", "above"),
        (r"(profit|loss|selling|purchase|total income|total expense)\s+(below|less than|under)\s+(\d+)", "below"),
        (r"(profit|loss|selling|purchase|total income|total expense)\s+equal\s+to\s+(\d+)", "equal")
    ]

    for pattern, op in patterns:
        match = re.search(pattern, text)
        if match:
            field = match.group(1)
            value = float(match.group(3))
            return field, op, value

    return None, None, None


if __name__ == "__main__":
    try:
        while True:
            speech_text = recognize_speech()
            if speech_text is None:
                continue

            # 1. Company name match
            if "company" in speech_text:
                company_results = search_company_by_name(speech_text)
                if company_results:
                    for idx, item in enumerate(company_results, start=1):
                        print(f"{idx}.")
                        for key in FIELDS:
                            print(f"{key:<16} {item.get(key, '')}")
                        print("-" * 40)
                    continue
                else:
                    print("❓ Company not found.")
                    continue

            # 2. Field + condition match (e.g., profit above 100)
            field, op, value = parse_condition_query(speech_text)
            if field and op:
                results = filter_data_by_condition(field, op, value)
                if results:
                    for idx, item in enumerate(results, start=1):
                        print(f"{idx}.")
                        for key in FIELDS:
                            print(f"{key:<16} {item.get(key, '')}")
                        print("-" * 40)
                else:
                    print("❓ No matching items found for condition.")
                continue

            # 3. Fallback: Top-N sorting
            try:
                sort_field = match_field_with_minilm(speech_text)
                numbers = re.findall(r'\d+', speech_text)
                top_n = int(numbers[0]) if numbers else 3

                results = sorted(data, key=lambda x: get_numeric_value(x, sort_field), reverse=True)[:top_n]

                if results:
                    for idx, item in enumerate(results, start=1):
                        print(f"{idx}.")
                        for key in FIELDS:
                            print(f"{key:<16} {item.get(key, '')}")
                        print("-" * 40)
                else:
                    print("❓ No matching items found.")
            except Exception as e:
                print(f"⚠️ Error: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Program stopped by user.")
