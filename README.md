# updated_fulfledge_prism
twin.py
Main recommendation logic.
Handles user profile loading and generates personalized recommendations using LLMs.

generate_questions.py
Question generation logic.
Extracts schema fields, generates conversational questions, ranks them by impact, and assigns them into tiers (Tier 1, 2, 3).

firebase_db.py
Firebase client setup.
Initializes and returns the Firebase Firestore client for reading/writing data.

app.py
Main interview logic and app UI.
Controls Streamlit interface, manages session state, loads questions, and drives the interview flow.
