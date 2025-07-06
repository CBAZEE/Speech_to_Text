import os, queue, sys, json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

model_path = r"D:\Speech to text\vosk-model-small-en-us-0.15"

if not os.path.exists(model_path):
    print("❌ Model not found."); sys.exit(1)

model = Model(model_path)
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

def on_shop():
    print("🛍️ 'shop' detected! Running action...")

print("🎤 Say 'shop' to trigger action...")

try:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "")
                print("📝 You said:", text)
                if "shop" in text.lower():
                    on_shop()
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                print("...", partial, end="\r")

except KeyboardInterrupt:
    print("\n👋 Exiting.")
except Exception as e:
    print(f"❌ Error: {e}")