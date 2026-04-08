import pickle, numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model/emotion_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

tests = [
    "i love you so much",
    "i am in love with you",
    "you make me feel so loved and special",
    "i miss you and i love you deeply",
    "my heart beats for you i adore you",
    "i feel so loved and appreciated today",
    "i have a crush on someone and i cant stop thinking about them",
    "she makes me feel butterflies in my stomach",
    "i am falling in love and it feels amazing",
    "you are the love of my life and i cant live without you",
    "i feel so much affection and warmth for my partner",
    "i told her i love her and she said it back",
    "being with you feels like home i love you more than words",
    "i just want to hold you and never let go",
    "thinking about you makes my heart so warm and full of love",
]

for text in tests:
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=50, padding="pre", truncating="pre")
    p = model.predict(pad, verbose=0)[0]
    em = EMOTIONS[np.argmax(p)]
    conf = p[np.argmax(p)]
    love_score = p[2]
    print(f"{'LOVE' if em=='love' else '    '} [{em:8s} {conf:.3f}] love={love_score:.3f} | {text}")
