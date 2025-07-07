# firebase.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# function to initiate firebase client
def get_db() -> firestore.Client:
    """Initialize and return Firestore client."""
    config = st.secrets["firebase"]
    cred = credentials.Certificate(dict(config))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()
