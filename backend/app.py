"""
Emotional Journaling API — Flask Backend
==========================================
Loads a trained Bidirectional LSTM model and tokenizer to predict
emotions from journal text.  Stores results in Supabase.
"""

import os
import re
import pickle
import string
from datetime import datetime, timedelta, timezone

import emoji
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Configuration ───────────────────────────────────────────────────────────

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model", "emotion_model.h5")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "model", "tokenizer.pkl")

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MAX_LEN  = 50          # must match training maxlen (from model input shape)

# ─── Init Flask ──────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, supports_credentials=True)

# ─── Load ML Model & Tokenizer ──────────────────────────────────────────────

print("⏳  Loading emotion model …")
model = load_model(MODEL_PATH)
print("✅  Model loaded.")

print("⏳  Loading tokenizer …")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("✅  Tokenizer loaded.")

# ─── Supabase Client ────────────────────────────────────────────────────────

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Helper Functions ────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean a raw text string before tokenisation."""
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))   # 😊 → smiling_face
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str):
    """Split text into sentences using punctuation boundaries."""
    # Split on sentence-ending punctuation, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_by_sentences(text: str, max_words: int = MAX_LEN):
    """
    Smart chunking: split by sentences, then group sentences
    into chunks that fit within max_words.
    If the text is short enough, return it as a single chunk.
    """
    words = text.split()

    # If text fits in one prediction, don't chunk at all
    if len(words) <= max_words:
        return [text]

    sentences = split_into_sentences(text)

    # If no sentence boundaries found, fall back to word-based chunks
    if len(sentences) <= 1:
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])
            if chunk:
                chunks.append(chunk)
        return chunks

    # Group sentences into chunks that stay under max_words
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_words = len(sentence.split())
        if current_len + sent_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sent_words
        else:
            current_chunk.append(sentence)
            current_len += sent_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Keywords that strongly indicate the "love" emotion
LOVE_KEYWORDS = {
    "love", "loved", "loving", "adore", "adored", "cherish", "cherished",
    "darling", "sweetheart", "romance", "romantic", "affection", "affectionate",
    "devoted", "devotion", "passion", "passionate", "heart", "soulmate",
    "beloved", "embrace", "kiss", "kisses", "hugs", "cuddle", "tender",
    "warmth", "caring", "deeply", "forever", "together", "partner",
}

LOVE_BOOST = 0.35  # how much to boost "love" score when keywords are present


def apply_love_boost(text: str, scores: dict) -> dict:
    """
    The model often confuses 'love' with 'joy'. If the text contains
    love-related keywords, boost the love score proportionally.
    """
    words = set(text.lower().split())
    love_word_count = len(words & LOVE_KEYWORDS)

    if love_word_count == 0:
        return scores

    # Scale boost by number of love words found (capped at 3)
    boost = LOVE_BOOST * min(love_word_count, 3)
    boosted = dict(scores)
    boosted["love"] = boosted.get("love", 0) + boost

    return boosted


def predict_emotion(text: str):
    """
    Run the full prediction pipeline on *text*.

    Returns
    -------
    dict with keys: emotion, confidence, chunk_emotions
    """
    # Split into sentences BEFORE preprocessing removes punctuation
    # (punctuation like . ? ! is needed for sentence boundary detection)
    chunks = chunk_text_by_sentences(text.lower())

    # Preprocess each chunk individually (remove punctuation, emojis, etc.)
    chunks = [preprocess_text(c) for c in chunks]
    chunks = [c for c in chunks if c.strip()]  # remove empty chunks

    chunk_emotions = []

    for chunk in chunks:
        seq = tokenizer.texts_to_sequences([chunk])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
        pred = model.predict(pad, verbose=0)[0]          # shape (6,)
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])
        chunk_emotions.append({
            "text": chunk,
            "emotion": EMOTIONS[idx],
            "confidence": round(conf, 4),
            "scores": {EMOTIONS[i]: round(float(pred[i]), 4) for i in range(len(EMOTIONS))}
        })

    # ── Weighted aggregation across chunks ────────────────────────────────
    total_scores = {e: 0.0 for e in EMOTIONS}
    for ce in chunk_emotions:
        for emotion, score in ce["scores"].items():
            total_scores[emotion] += score

    # ── Love keyword boosting ─────────────────────────────────────────────
    # The model's training data has weak "love" representation, causing
    # confusion with "joy". Boost love score when love keywords are present.
    total_scores = apply_love_boost(text.lower(), total_scores)

    final_emotion = max(total_scores, key=total_scores.get)
    total_sum = sum(total_scores.values()) or 1
    final_confidence = round(total_scores[final_emotion] / total_sum, 4)

    return {
        "emotion": final_emotion,
        "confidence": final_confidence,
        "chunk_emotions": chunk_emotions,
    }


# ─── Supabase helper: create user-scoped client ─────────────────────────────

def get_user_client(access_token: str) -> Client:
    """Return a Supabase client that acts on behalf of the authenticated user."""
    from supabase import create_client as _cc
    client = _cc(SUPABASE_URL, SUPABASE_KEY)
    client.postgrest.auth(access_token)
    return client

def extract_token(req):
    """Extract Bearer token from Authorization header."""
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Serve Frontend ──────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# ─── POST /api/predict ───────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept journal text, predict emotion, store in Supabase."""
    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    user_id = data.get("user_id")
    token   = extract_token(request)

    if not user_id or not token:
        return jsonify({"error": "Authentication required"}), 401

    # Run prediction
    result = predict_emotion(text)

    # Store in Supabase
    try:
        user_client = get_user_client(token)
        user_client.table("journal_entries").insert({
            "user_id":    user_id,
            "text":       text,
            "emotion":    result["emotion"],
            "confidence": result["confidence"],
        }).execute()
    except Exception as e:
        print(f"⚠️  DB insert error: {e}")
        # Still return the prediction even if storage fails
        result["db_warning"] = str(e)

    return jsonify(result), 200


# ─── GET /api/entries ─────────────────────────────────────────────────────────

@app.route("/api/entries", methods=["GET"])
def api_entries():
    """Fetch all journal entries for the authenticated user."""
    token   = extract_token(request)
    user_id = request.args.get("user_id")

    if not user_id or not token:
        return jsonify({"error": "Authentication required"}), 401

    try:
        user_client = get_user_client(token)
        resp = (
            user_client
            .table("journal_entries")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return jsonify(resp.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── DELETE /api/entries/<id> ─────────────────────────────────────────────────

@app.route("/api/entries/<entry_id>", methods=["DELETE"])
def api_delete_entry(entry_id):
    """Delete a specific journal entry."""
    token   = extract_token(request)
    user_id = request.args.get("user_id")

    if not user_id or not token:
        return jsonify({"error": "Authentication required"}), 401

    try:
        user_client = get_user_client(token)
        user_client.table("journal_entries").delete().eq("id", entry_id).eq("user_id", user_id).execute()
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── GET /api/insights ───────────────────────────────────────────────────────

@app.route("/api/insights", methods=["GET"])
def api_insights():
    """Compute derived insights from the user's entries."""
    token   = extract_token(request)
    user_id = request.args.get("user_id")

    if not user_id or not token:
        return jsonify({"error": "Authentication required"}), 401

    try:
        user_client = get_user_client(token)
        resp = (
            user_client
            .table("journal_entries")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        entries = resp.data
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not entries:
        return jsonify({
            "total_entries": 0,
            "most_frequent_emotion": None,
            "emotion_distribution": {},
            "streak": {"emotion": None, "days": 0, "type": None},
            "weekly_summary": None,
            "most_emotional_day": None,
        }), 200

    # ── Emotion distribution ──────────────────────────────────────────────
    emotion_counts = {}
    emotion_confidence_sum = {}
    for e in entries:
        em = e["emotion"]
        emotion_counts[em] = emotion_counts.get(em, 0) + 1
        emotion_confidence_sum[em] = emotion_confidence_sum.get(em, 0.0) + e.get("confidence", 0.0)
        
    # Break ties using cumulative confidence score
    most_frequent = max(emotion_counts, key=lambda k: (emotion_counts[k], emotion_confidence_sum[k]))

    # ── Streak calculation ────────────────────────────────────────────────
    positive = {"joy", "love", "surprise"}
    negative = {"sadness", "anger", "fear"}

    streak_type = None
    streak_days = 0
    if entries:
        first_emotion = entries[0]["emotion"]
        is_positive = first_emotion in positive
        streak_type = "positive" if is_positive else "negative"
        target_set = positive if is_positive else negative

        prev_date = None
        for e in entries:
            entry_date = e["created_at"][:10]
            if e["emotion"] not in target_set:
                break
            if prev_date and prev_date != entry_date:
                streak_days += 1
            elif prev_date is None:
                streak_days = 1
            prev_date = entry_date

    # ── Weekly summary ────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    weekly_entries = [
        e for e in entries
        if e["created_at"][:10] >= week_ago.strftime("%Y-%m-%d")
    ]
    weekly_emotions = {}
    weekly_confidence = {}
    for e in weekly_entries:
        em = e["emotion"]
        weekly_emotions[em] = weekly_emotions.get(em, 0) + 1
        weekly_confidence[em] = weekly_confidence.get(em, 0.0) + e.get("confidence", 0.0)
    weekly_top = max(weekly_emotions, key=lambda k: (weekly_emotions[k], weekly_confidence[k])) if weekly_emotions else None
    weekly_summary = f"You have been mostly feeling {weekly_top} this week." if weekly_top else None

    # ── Most emotional day ────────────────────────────────────────────────
    day_confidence = {}
    for e in entries:
        day = e["created_at"][:10]
        day_confidence[day] = day_confidence.get(day, 0) + e.get("confidence", 0)
    most_emotional_day = max(day_confidence, key=day_confidence.get) if day_confidence else None

    return jsonify({
        "total_entries": len(entries),
        "most_frequent_emotion": most_frequent,
        "emotion_distribution": emotion_counts,
        "streak": {
            "emotion": entries[0]["emotion"] if entries else None,
            "days": streak_days,
            "type": streak_type,
        },
        "weekly_summary": weekly_summary,
        "most_emotional_day": most_emotional_day,
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=5000)
