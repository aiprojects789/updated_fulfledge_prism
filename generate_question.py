import json
import re
import streamlit as st
from typing import List, Dict, Any
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from firebase_db import get_db

# initiating firebase client
db = get_db()

# checking document on firebase (testing purpose only)
def check_document_exists(collection_name: str, document_id: str) -> bool:
    doc_ref = db.collection(collection_name).document(document_id)
    return doc_ref.get().exists


# function to upload on firebase
def upload_dict_to_firestore(data: dict, collection_name: str, document_id: str) -> None:
    doc_ref = db.collection(collection_name).document(document_id)
    doc_ref.set(data)

# --- LLM Helpers --------------------------------------------------------------

# extracts the first JSON array from a string.
def extract_json_array(s: str) -> str:
    pattern = r'\[\s*(?:\{.*?\}\s*,?\s*)+\]'
    match = re.search(pattern, s, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM response")
    return match.group(0)

# recursively finds all dotted paths in a nested dictionary.
# where the corresponding value has a 'description', 'values', or 'value' key.
def get_concept_paths(data: dict, parent_key: str = '', sep: str = '.') -> List[str]:
    paths: List[str] = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            if 'description' in value or 'values' in value or 'value' in value:
                paths.append(new_key)
            else:
                for child_key, child_val in value.items():
                    if isinstance(child_val, dict):
                        paths.extend(get_concept_paths({child_key: child_val}, new_key, sep))
    return paths

# Given a dotted path (eg: "userContext.lifeStageNotes"), this function
# navigates through the nested dictionary and returns the 'description' field at that path.
def get_description_for_path(root: dict, dotted_path: str) -> str:
    parts = dotted_path.split('.') if dotted_path else []
    node = root
    for key in parts:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return node.get('description', '') if isinstance(node, dict) else ''


# uses the field path and its description to generate a conversational, open-ended question.
# Uses a ChatOpenAI model with a friendly system prompt to generate thoughtful questions.
def generate_single_question(field_path: str, intent_desc: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI assistant helping users build their personalized digital twin for better recommendations. Your tone should be warm, encouraging, and respectful."),
        ("user", f"""
You will be given a field from a JSON schema and a description explaining its intent.

Generate a conversational, open-ended question that:
- Feels like part of a friendly dialogue
- Offers light guidance or an example to help the user answer
- Is inclusive and privacy-aware (especially if the topic is sensitive)
- Does NOT include multiple-choice options or lists
- Sounds like something a thoughtful assistant would naturally ask

Use the field name and description to craft the question.

Example input:
Field Name: userContextAndLifestyle.lifeStageNotes  
Description: Capture the user’s current phase in life, such as studying, working, married, retired, etc.

Example output:
What stage of life are you currently in? Feel free to share if you’re studying, working, raising a family, or going through any major change right now.

Now generate a similar conversational question for:

Field Name: {field_path}  
Description: {intent_desc}
"""),
    ])
    model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=st.secrets['api']['key'])
    chain = prompt | model
    return chain.invoke({}).content.strip()


# uses GPT-4o to rank a list of question dictionaries by their impact (potential of extracting best insights for recommendation)
def rank_and_tier_with_gpt4o(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system = SystemMessagePromptTemplate.from_template(
        "You are an expert in personalization. Score each question 0–100 on impact and bucket into three tiers."
    )
    human = HumanMessagePromptTemplate.from_template(
        """Here is a JSON array of questions. Respond with ONLY a JSON array of the same objects,
each with added: impactScore (0–100) and tier (Tier 1/2/3), sorted by descending score.
```json
{questions}
```"""
    )
    prompt = ChatPromptTemplate.from_messages([system, human])
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=st.secrets['api']['key'])
    raw = llm(prompt.format_messages(questions=json.dumps(questions, indent=2)))
    return json.loads(extract_json_array(raw.content))




# enriches flat question data by extracting section/subsection and adding descriptions from schema.
# adds the following fields to each question:
#   - 'section': top-level key from the field path
#   - 'subsection': the rest of the field path
#   - 'description': text from schema at that path
#   - 'qest': a placeholder flag, set to 'pending'
def enrich_questions(flat_questions: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for q in flat_questions:
        full = q['field']
        section, *rest = full.split('.', 1)
        subsection = rest[0] if rest else ''
        description = get_description_for_path(schema.get(section, {}), subsection)
        enriched.append({**q, 'section': section, 'subsection': subsection, 'description': description, 'qest': 'pending'})
    return enriched




# groups a flat list of enriched questions into tiers based on the 'tier' key.
# Returns a dictionary with keys: tier1, tier2, tier3 — each having:
#   - 'status': status flag for that tier (default = 'in_process')
#   - 'questions': list of questions in that tier
def wrap_questions_by_tier(flat_questions: List[Dict[str, Any]], status: str = 'in_process') -> Dict[str, Any]:
    grouped = defaultdict(list)
    for q in flat_questions:
        grouped[q['tier']].append(q)
    return {
        'tier1': {'status': status, 'questions': grouped.get('Tier 1', [])},
        'tier2': {'status': status, 'questions': grouped.get('Tier 2', [])},
        'tier3': {'status': status, 'questions': grouped.get('Tier 3', [])},
    }

# --- Streamlit App -----------------------------------------------------------
st.title('📝 Digital Twin Interview Question Generator')

# Profile JSON Uploader
uploaded = st.file_uploader('Upload profile JSON', type=['json'], key='uploader_profile')
json_data = {}
if uploaded:
    json_data = json.load(uploaded)
    doc_id = uploaded.name
    if not check_document_exists('user_collection', doc_id):
        upload_dict_to_firestore(json_data, 'user_collection', doc_id)
        st.success(f"Uploaded profile as '{doc_id}'")
    else:
        st.info(f"Profile '{doc_id}' exists. Skipped upload.")
else:
    st.info('Please upload a profile JSON.')

# Section Selector
section = st.radio('Select section:', ['General Profile', 'Recommendation Profile', 'Simulation Preferences'], key='section_selector')
category = None
if section == 'Recommendation Profile' and uploaded:
    cats = list(json_data.get('recommendationProfiles', {}).keys())
    category = st.selectbox('Category', cats, key='rec_category')

# Generate Button
if st.button('Generate Questions', key='btn_generate'):
    if not uploaded:
        st.warning('Upload a profile JSON first.')
    else:
        # Generate questions
        with st.spinner("Generating Questions...."):
            sect_key = section.lower().replace(' ', '')
            flat: List[Dict[str, Any]] = []
            if sect_key == 'generalprofile':
                for p in get_concept_paths(json_data.get('generalprofile', {})):
                    path = f"generalprofile.{p}"
                    flat.append({'field': path, 'question': generate_single_question(path, get_description_for_path(json_data['generalprofile'], p))})
            elif sect_key == 'recommendationprofile' and category:
                for p in get_concept_paths(json_data['recommendationProfiles'].get(category, {})):
                    path = f"recommendationProfiles.{category}.{p}"
                    flat.append({'field': path, 'question': generate_single_question(path, get_description_for_path(json_data['recommendationProfiles'][category], p))})
            else:
                for p in get_concept_paths(json_data.get('simulationPreferences', {})):
                    path = f"simulationPreferences.{p}"
                    flat.append({'field': path, 'question': generate_single_question(path, '')})

        ranked = rank_and_tier_with_gpt4o(flat)
        enriched = enrich_questions(ranked, json_data)
        wrapped = wrap_questions_by_tier(enriched)

        file_name = f"{('general' if sect_key=='generalprofile' else category if category else 'simulation')}_tiered_questions.json"

        upload_dict_to_firestore(wrapped, 'question_collection', file_name)
        st.success(f"Uploaded questions as '{file_name}'")

        st.json(wrapped)
        st.download_button('Download Questions', data=json.dumps(wrapped, indent=2), file_name=file_name, key='download_questions')


# --- Firebase File Management in Sidebar (Most of these are for testing purpose)

# checking if document exits on firebase
def check_document_exists(collection: str, doc_id: str) -> bool:
    return db.collection(collection).document(doc_id).get().exists

# uploading document on firebase
def upload_dict_to_firestore(data: dict, collection: str, doc_id: str) -> None:
    db.collection(collection).document(doc_id).set(data)

# listing firebase documents
def list_document_ids(collection: str) -> List[str]:
    return [doc.id for doc in db.collection(collection).list_documents()]

# deleting document from firebase
def delete_document(collection: str, doc_id: str) -> None:
    db.collection(collection).document(doc_id).delete()

# downloading document from firebase
def download_document(collection: str, doc_id: str) -> dict:
    doc = db.collection(collection).document(doc_id).get()
    if doc.exists:
        return doc.to_dict()
    else:
        return {}


st.sidebar.subheader("Firebase File Browser")
col = st.sidebar.selectbox("Select Collection to browse:", ["user_collection", "question_collection"], key='browser_col')
if st.sidebar.button("List Files", key='btn_list'):
    docs = list_document_ids(col)
    st.sidebar.write(f"Documents in {col}:")
    for d in docs:
        st.sidebar.write(f"- {d}")

st.sidebar.write("---")

# sidebar (functions that are only for testing purpose)
del_col = st.sidebar.selectbox("Select Collection to delete from:", ["user_collection", "question_collection"], key='del_col')
docs_to_del = list_document_ids(del_col)

# deleting from firebase logic
delete_id = st.sidebar.selectbox("Select Document to Delete:", docs_to_del, key='del_select')
if st.sidebar.button("Delete File", key='btn_delete'):
    delete_document(del_col, delete_id)
    st.sidebar.success(f"Deleted '{delete_id}' from {del_col}.")

# Download file from firebase
if st.sidebar.button("Download File", key='btn_download'):
    doc_data = download_document(del_col, delete_id)
    if doc_data:
        json_data = json.dumps(doc_data, indent=2)
        st.sidebar.download_button(
            label=f"Download '{delete_id}.json'",
            data=json_data,
            file_name=f"{delete_id}.json",
            mime="application/json"
        )
    else:
        st.sidebar.error(f"No document found with ID '{delete_id}' in '{del_col}'.")
