import csv
import random
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from googletrans import Translator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class FarmerChatbot:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_faqs("farmer_dataset.csv")
        self._init_safety_filters()
        self.translator = Translator()
        self.compliment_given = False

        # Load trained crop disease prediction model
        self.model = load_model("plant_disease_model.h5")

        # Load labels
        self.class_labels = self._load_class_labels("labels.txt")

        # Load cures from dataset
        self.disease_cures = self._load_disease_cures("farmer_dataset.csv")

    def _load_class_labels(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filepath}. Please create it with class names.")
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_disease_cures(self, filepath):
        cures = {}
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['patterns']:
                    cures[row['patterns']] = row['response'].split(',')[0].strip()
        return cures

    def _load_faqs(self, filepath):
        self.questions, self.answers_en, self.answers_ml = [], [], []
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['patterns']:
                    patterns = [p.strip().lower() for p in row['patterns'].split('|')]
                    for p in patterns:
                        self.questions.append(p)
                        if ',' in row['response']:
                            parts = row['response'].split(',', 1)
                            self.answers_en.append(parts[0].strip())
                            self.answers_ml.append(parts[1].strip())
                        else:
                            self.answers_en.append(row['response'].strip())
                            self.answers_ml.append(row['response'].strip())

        embeddings = self.embedder.encode(self.questions)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _init_safety_filters(self):
        self.inappropriate = re.compile(
            r'\b(fuck|shit|damn|bitch|asshole)\b|love you|hate you',
            re.IGNORECASE
        )
        self.compliments = re.compile(
            r'\b(thanks|thank you|awesome|great|good job|helpful|appreciate)\b',
            re.IGNORECASE
        )

    def _find_faq_match(self, query, threshold=0.72):
        query_embed = self.embedder.encode([query.lower()])
        faiss.normalize_L2(query_embed)
        scores, indices = self.index.search(query_embed, k=3)
        if scores[0][0] > threshold:
            return indices[0][0]
        for i in range(3):
            if scores[0][i] > threshold - 0.1 * (i + 1):
                return indices[0][i]
        return None

    def _handle_compliment(self, query):
        if self.compliments.search(query):
            self.compliment_given = True
            return random.choice(["Happy to help! ", "Glad I could assist! ", "You're welcome! "])
        return ""

    def detect_language(self, text):
        try:
            lang = self.translator.detect(text).lang
            return "ml" if lang == "ml" else "en"
        except:
            return "en"

    def predict_disease(self, image_path, lang="en"):
        """Predict disease from image and return cure"""
        img = image.load_img(image_path, target_size=(128, 128))  # match your training size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(preds)]

        # Find cure from CSV mapping
        cure = self.disease_cures.get(predicted_class, "No cure information available.")

        if lang == "ml":
            predicted_class_translated = self.translator.translate(predicted_class, dest="ml").text
            cure_translated = self.translator.translate(cure, dest="ml").text
            return f"Prediction: {predicted_class_translated}\nCure: {cure_translated}"
        else:
            return f"Prediction: {predicted_class}\nCure: {cure}"

    def get_response(self, user_input):
        if self.inappropriate.search(user_input):
            return "Please keep queries respectful."

        polite_response = self._handle_compliment(user_input)
        user_lang = self.detect_language(user_input)

        cleaned_input = re.sub(r'\b(wat|u|r|b4)\b', lambda m: {
            'wat': 'what', 'u': 'you', 'r': 'are', 'b4': 'before'
        }[m.group().lower()], user_input, flags=re.IGNORECASE)

        match_idx = self._find_faq_match(cleaned_input)
        if match_idx is not None:
            return polite_response + (self.answers_ml[match_idx] if user_lang == "ml" else self.answers_en[match_idx])

        fallback = "Could you rephrase? I handle crop, soil, and disease queries."
        return polite_response + (self.translator.translate(fallback, dest="ml").text if user_lang == "ml" else fallback)


# Initialize chatbot instance
bot_instance = FarmerChatbot()

def get_response(user_input):
    return bot_instance.get_response(user_input)

def predict_disease(image_path, lang="en"):
    return bot_instance.predict_disease(image_path, lang)
