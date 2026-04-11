# MoodScribe 📝🔮

MoodScribe is an AI-powered emotional journaling application that tracks your mental well-being securely. It uses a custom deep learning model and intelligent sentence-chunking to analyze the emotions in your daily entries, visualizing your mood trends, patterns, and streaks on an interactive dashboard.

## 🚀 Features
- **AI Emotion Detection**: Analyzes text to detect 6 core emotions (Joy, Sadness, Anger, Fear, Surprise, Love).
- **Smart Sentence Chunking**: Breaks down long journal entries into individual sentences for a granular, "chunk-level" emotional analysis.
- **Interactive Dashboard**: View your journal history, see your overall "Typical Emotion", track emotion streaks, and view dynamic charts representing your mood distribution.
- **Secure Authentication & Storage**: Powered securely by Supabase for user authentication and Postgres database storage.
- **Toggle Privacy**: Secure login/signup system with password visibility toggles.
- **Entry Management**: Click to view full expanded entries via stylish modals, or permanently delete specific journal entries.

## 🛠️ Technology Stack
- **Frontend**: Vanilla HTML5, CSS3 (with custom variables, modern gradients, & glassmorphism), JavaScript
- **Backend Framework**: Python (Flask)
- **Machine Learning**: TensorFlow / Keras (Bidirectional LSTM neural network)
- **Database & Auth**: Supabase (PostgreSQL)
- **Data Visualisation**: Chart.js (for pie and timeline charts)

---

## 💻 Getting Started (Local Development)

### 1. Prerequisites
- **Python 3.8+** installed on your machine.
- A **Supabase** account to store user data.

### 2. Set Up Supabase
1. Create a new Supabase project.
2. In the Supabase SQL Editor, run the following to create the required table:
    ```sql
    CREATE TABLE journal_entries (
      id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
      user_id uuid REFERENCES auth.users NOT NULL,
      text text NOT NULL,
      emotion text NOT NULL,
      confidence real NOT NULL,
      created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
    );
    ```
3. Establish Row Level Security (RLS) policies for user privacy so they can only interact with their own entries:
    ```sql
    ALTER TABLE journal_entries ENABLE ROW LEVEL SECURITY;

    CREATE POLICY "Users can insert their own entries" ON journal_entries
      FOR INSERT WITH CHECK (auth.uid() = user_id);

    CREATE POLICY "Users can view their own entries" ON journal_entries
      FOR SELECT USING (auth.uid() = user_id);
      
    CREATE POLICY "Users can delete own entries" ON journal_entries
      FOR DELETE USING (auth.uid() = user_id);
    ```

### 3. Environment Variables
Create a file named `.env` in the root folder (or `backend/` folder) and add your Supabase credentials:
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### 4. Installation
Install all backend dependencies via pip:
```bash
cd backend
pip install -r requirements.txt
```

### 5. Running the App
Start the Flask development server:
```bash
python app.py
```
Once the sever has initialized the machine learning model, go to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🧠 Model Details
The AI backend utilizes an `.h5` TensorFlow BiLSTM model accompanied by a custom Tokenizer. To combat class imbalance found in standard emotion datasets (where 'Love' is frequently misclassified as 'Joy'), the prediction pipeline features keyword-heuristic boosting to enhance contextual accuracy. The analyzer parses sequences, extracts contextual meaning natively, handles edge cases around punctuation layout, and breaks entries mathematically to deduce an overwhelming emotional verdict.

## 🐳 Docker Deployment (Optional)
If you wish to containerize this app, a lightweight `Dockerfile` and `.dockerignore` are provided:
1. Build image: `docker build -t moodscribe-app .`
2. Run image: `docker run -d -p 5000:5000 --name moodscribe moodscribe-app`
