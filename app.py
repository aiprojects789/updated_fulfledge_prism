import streamlit as st
# from upload_to_db import upload_json_data_to_firestore, document_exists
import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from twin import load_user_profile, generate_recommendations
# from quest_generate import get_pending_questions_by_field
# from quest_generate import get_next_pending_question
from firebase_db import get_db


# Tiered Interview Class (main interview logic and flow)
class TieredInterviewAgent:
    def __init__(self, db, openai_key):
        self.db = db
        self.openai_key = openai_key
        self.current_tier_idx = 0
        self.current_phase = 'general'
        self.current_q_idx = 0
        self.tier_keys = []
        self.general_questions = {}
        self.category_questions = {}
        self.profile_structure = {}

        # fetching category document id for category question files (based on selection in UI)
        selected = st.session_state.get('Selected_category', 'Movies').lower()
        cat_map = {
                    'movies': 'moviesAndTV_tiered_questions.json',
                    'food':   'foodAndDining_tiered_questions.json',
                    'travel': 'travel_tiered_questions.json'
                }
        self.cat_doc_id = cat_map.get(selected, 'moviesAndTV_tiered_questions.json')

        self.load_data()

    # function to load data from firebase
    def load_data(self):
        try:
            gen_doc = self.db.collection("question_collection").document("general_tiered_questions.json").get()
            self.general_questions = gen_doc.to_dict() if gen_doc.exists else {}
            if not gen_doc.exists:
                st.warning("general_tiered_questions.json not found in Firestore")

            # Load category using precomputed ID
            cat_doc = self.db.collection("question_collection").document(self.cat_doc_id).get()
            self.category_questions = cat_doc.to_dict() if cat_doc.exists else {}
            if not cat_doc.exists:
                st.warning(f"{self.cat_doc_id} not found in Firestore")

            # Load profile structure
            profile_doc = self.db.collection("user_collection").document("profile_strcuture.json").get() ## profile_strcuture.json this will be unique for every user
            self.profile_structure = profile_doc.to_dict() if profile_doc.exists else {}
            if not profile_doc.exists:
                st.warning("profile_strcuture.json not found in Firestore")

            # Extract tier keys
            if self.general_questions:
                self.tier_keys = sorted(
                    [k for k in self.general_questions.keys() if k.startswith('tier')],
                    key=lambda x: int(x.replace('tier', ''))
                )
            else:
                self.tier_keys = []
                st.error("No tier data found in general questions")

        except Exception as e:
            st.error(f"Failed to load data: {e}")
            self.general_questions = {}
            self.category_questions = {}
            self.profile_structure = {}
            self.tier_keys = []

        self.pick_up_where_left_off()

    # logic for interviewing resuming
    def pick_up_where_left_off(self):
        """
        Find the first tier that is not completed and set it to in_process.
        If all tiers are completed, mark the interview as complete.
        """
        for idx, tier_key in enumerate(self.tier_keys):
            status = self.general_questions.get(tier_key, {}).get('status', '')
            if status == 'completed':
                continue

            # Resuming here
            self.current_tier_idx = idx
            if status != 'in_process':
                self.general_questions[tier_key]['status'] = 'in_process'
            return

        # If no tier left, mark interview complete by moving index past last
        self.current_tier_idx = len(self.tier_keys)

    # function for referencing current tier
    def get_current_tier_key(self):
        """Get current tier key"""
        if self.current_tier_idx < len(self.tier_keys):
            return self.tier_keys[self.current_tier_idx]
        return None

    # function for fetching pending questions
    def get_pending_questions(self, dataset, tier_key):
        """Get pending questions for a specific tier and dataset"""
        if not dataset or not tier_key or tier_key not in dataset:
            return []
            
        tier = dataset.get(tier_key, {})
        
        # For general questions, with respect the tier status
        if dataset == self.general_questions:
            tier_status = tier.get('status', '')
            if tier_status != 'in_process' and tier_status != '':
                return []
        
        # Return pending questions 
        questions = tier.get('questions', [])
        if not isinstance(questions, list):
            return []
            
        return [q for q in questions if isinstance(q, dict) and q.get('qest') == 'pending']
    
    # function to get the current question from pending questions 
    def get_current_question(self):
        """Get the current question to be asked"""
        tier_key = self.get_current_tier_key()
        if not tier_key:
            return None
            
        if self.current_phase == 'general':
            pending = self.get_pending_questions(self.general_questions, tier_key)
            if pending and 0 <= self.current_q_idx < len(pending):
                question_data = pending[self.current_q_idx]
                return {
                    'question': question_data.get('question', ''),
                    'field': question_data.get('field', ''),
                    'phase': 'general',
                    'tier': tier_key
                }
        elif self.current_phase == 'category':
            pending = self.get_pending_questions(self.category_questions, tier_key)
            if pending and 0 <= self.current_q_idx < len(pending):
                question_data = pending[self.current_q_idx]
                return {
                    'question': question_data.get('question', ''),
                    'field': question_data.get('field', ''),
                    'phase': 'category',
                    'tier': tier_key
                }
        
        return None
    
    # function to regenerate and add llm generated conversational style
    def regenerate_question_with_motivation(self, next_question: str, user_response: str = None) -> str:
        """
        Generate a conversational follow-up by acknowledging the user's response, then smoothly introducing the next question.
        The goal is to create a natural, friendly transition that weaves in encouragement or personal acknowledgment.

        Returns a conversational-style message that leads into the next question.
        """
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o-mini",
            openai_api_key=self.openai_key
        )

        # Build messages
        messages = [
            SystemMessage(content=(
                "You are a friendly, engaging interviewer having a casual, supportive conversation. "
                "When provided with a user's previous response and a next question, create a natural, conversational transition. "
                "Acknowledge or positively reflect on the user's response, and then smoothly ask the next question. "
                "Keep the tone friendly, curious, and encouraging, and avoid robotic phrasing. "
                "Do not rigidly repeat the question; weave it naturally into your words."
            ))
        ]

        # Construct the prompt
        prompt = f"Next question: {next_question}\n"
        if user_response:
            prompt += f"User's previous response: {user_response}\n"
        prompt += (
            "Please write a natural, conversational transition that acknowledges the user's response "
            "and leads into the next question. Keep it warm, curious, and supportive."
        )

        messages.append(HumanMessage(content=prompt))

        # Get LLM response
        response = llm(messages)
        return response.content.strip()

    # function to submit answer to profile at firebase
    def submit_answer(self, answer):
        """Submit answer and update profile structure"""
        tier_key = self.get_current_tier_key()
        if not tier_key:
            return False
            
        current_q = self.get_current_question()
        if not current_q:
            return False
        
        # Update the question status in the appropriate dataset
        if self.current_phase == 'general':
            dataset = self.general_questions
        else:
            dataset = self.category_questions
            
        pending = self.get_pending_questions(dataset, tier_key)
        if pending and 0 <= self.current_q_idx < len(pending):
            # Find the question in the original dataset and mark as answered
            question_to_update = pending[self.current_q_idx]
            question_text = question_to_update.get('question', '')
            
            # Safely find and update the question
            tier_questions = dataset.get(tier_key, {}).get('questions', [])
            for q in tier_questions:
                if q.get('question') == question_text:
                    q['qest'] = 'answered'
                    break
            
            # Update profile structure with the answer
            field_path = current_q.get('field', '')
            if field_path:
                self.update_profile_structure(field_path, answer)
            
            # Move to next question or phase
            self.advance_to_next()
            
            return True
        
        return False
    
    # function to add the answer in profile_structure file (inside "value")
    def update_profile_structure(self, field_path, answer):
        """Update profile structure with the answer"""
        if not field_path or not isinstance(field_path, str):
            return
            
        # Navigate to the correct field in profile structure
        keys = field_path.split('.')
        current = self.profile_structure
        
        try:
            # Navigate to the parent of the target field
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Update the value
            final_key = keys[-1]
            if final_key in current and isinstance(current[final_key], dict):
                current[final_key]['value'] = answer
            else:
                # Create the structure if it doesn't exist
                if final_key not in current:
                    current[final_key] = {}
                if isinstance(current[final_key], dict):
                    current[final_key]['value'] = answer
        except (KeyError, TypeError, AttributeError) as e:
            st.error(f"Error updating profile structure for field '{field_path}': {e}")
            return
    

    # function to advancing between general questions and category questions, also manages to move to next tiers once a tier completed
    def advance_to_next(self):
        """Advance to next question, phase, or tier"""
        tier_key = self.get_current_tier_key()
        if not tier_key:
            return
        
        if self.current_phase == 'general':
            pending = self.get_pending_questions(self.general_questions, tier_key)
            if pending and self.current_q_idx + 1 < len(pending):
                self.current_q_idx += 1
            else:
                # Move to category phase
                self.current_phase = 'category'
                self.current_q_idx = 0
                
                # Check if there are category questions for this tier
                cat_pending = self.get_pending_questions(self.category_questions, tier_key)
                if not cat_pending:
                    # No category questions, complete tier and move to next
                    self.complete_current_tier()
                    self.advance_to_next_tier()
        
        elif self.current_phase == 'category':
            pending = self.get_pending_questions(self.category_questions, tier_key)
            if pending and self.current_q_idx + 1 < len(pending):
                self.current_q_idx += 1
            else:
                # Complete current tier and move to next
                self.complete_current_tier()
                self.advance_to_next_tier()
    
    # function to mark tier status as "completed"
    def complete_current_tier(self):
        """Mark current tier as completed"""
        tier_key = self.get_current_tier_key()
        if tier_key:
            # Mark general tier as completed
            if tier_key in self.general_questions:
                self.general_questions[tier_key]['status'] = 'completed'
            
            # Mark category tier as completed if it exists
            if tier_key in self.category_questions:
                self.category_questions[tier_key]['status'] = 'completed'
    
    # function to move to next tier
    def advance_to_next_tier(self):
        """Move to the next tier"""
        if self.current_tier_idx + 1 < len(self.tier_keys):
            self.current_tier_idx += 1
            self.current_phase = 'general'
            self.current_q_idx = 0
            
            # Set next tier status to 'in_process'
            next_tier_key = self.get_current_tier_key()
            if next_tier_key and next_tier_key in self.general_questions:
                self.general_questions[next_tier_key]['status'] = 'in_process'
        else:
            # No more tiers, mark as complete
            self.current_tier_idx = len(self.tier_keys)
    
    # function to check interview completion status
    def is_complete(self):
        """Check if interview is complete"""
        return self.current_tier_idx >= len(self.tier_keys)
    
    # function to save profiles progress to firebase
    def save_to_firestore(self):
        """Save all data back to Firestore"""
        try:
            # Save profile structure
            self.db.collection("user_collection").document("profile_strcuture.json").set(self.profile_structure)  # profile_strcuture.json this will be user specific
            
            # Save general questions
            self.db.collection("question_collection").document("general_tiered_questions.json").set(self.general_questions)
            
            # Save category questions
            self.db.collection("question_collection").document(self.cat_doc_id).set(self.category_questions)
            
            return True
        except Exception as e:
            st.error(f"Failed to save to Firestore: {e}")
            return False
        
    

# --- CONFIG & FIREBASE SETUP ---
openai_key = st.secrets["api"]["key"]

db = get_db()

# --- PAGE TITLE ---
st.markdown(
    '<h1 class="title" style="text-align: center; font-size: 80px; color: #E041B1;">Prism</h1>',
    unsafe_allow_html=True
)

# --- SESSION STATE FLAGS ---
if "show_recs" not in st.session_state:
    st.session_state.show_recs = False
if "interview_messages" not in st.session_state:
    st.session_state.interview_messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.image("logo trans.png", width=200)

    # custom button styling
    st.markdown("""
    <style>
      .stButton button {
        background-color: #2c2c2e;
        color: white;
        font-size: 16px;
        padding: 8px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
      }
      .stButton button:hover { background-color: #95A5A6; }
      .stButton button:active { background-color: #BDC3C7; }
    </style>""", unsafe_allow_html=True)
    
    # Category selection
    st.sidebar.selectbox(
        "Select Category:",
        ["Movies", "Food", "Travel"],
        key='Selected_category',
        on_change=lambda: st.session_state.pop('tiered_interview_agent', None)
    )
    
    # Switch to Recommendation mode
    if st.sidebar.button("Get Recommendation"):
        st.session_state.show_recs = True

    # Reset everything (for debug and testing purpose only)
    if st.sidebar.button("Reset Interview"):
        for key in ["interview_messages", "tiered_interview_agent", "show_recs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- MAIN PAGE CONTENT ---
st.markdown("---")

if "show_recs" in st.session_state and st.session_state.show_recs:
    #  RECOMMENDATION UI
    st.subheader("Personalized Recommendations")

    # Load profile once
    if "profile_loaded" not in st.session_state:
        st.session_state.profile_loaded = load_user_profile()

    if not st.session_state.profile_loaded:
        st.error("No profile found—complete the interview first.")
    else:
        # ask for query
        query = st.text_input("What would you like recommendations for?", key="rec_query")
        if st.button("Generate Recommendations"):
            with st.spinner("Thinking…"):
                try:
                    recs_json = generate_recommendations(st.session_state.profile_loaded, query)
                    recs = json.loads(recs_json)

                    # Normalize to list
                    if isinstance(recs, dict):
                        if "recommendations" in recs and isinstance(recs["recommendations"], list):
                            recs = recs["recommendations"]
                        else:
                            recs = [recs]

                    if not isinstance(recs, list):
                        st.error("❌ Unexpected response format – expected a list of recommendations.")
                    else:
                        for i, item in enumerate(recs, 1):
                            title = item.get("title", "<no title>")
                            reason = item.get("reason", "<no reason>")
                            st.markdown(f"**{i}. {title}**")
                            st.write(reason)

                except Exception as err:
                    st.error(f"Failed: {err}")

    # "back button to arrive back at interview page
    if st.button("← Back to Interview"):
        st.session_state.show_recs = False
        st.rerun()

else:
    # TIERED INTERVIEW UI 
    st.subheader("Prism Tiered Interview")
    st.write("Complete the tiered interview to build your comprehensive profile:")

    # --- Initialize Tiered Interview Agent ---
    if "tiered_interview_agent" not in st.session_state:
        agent = TieredInterviewAgent(db, openai_key)
        
        if not agent.is_complete():
            st.session_state.tiered_interview_agent = agent
            current_q = agent.get_current_question()
            
            if current_q:
                st.session_state.interview_messages = [
                    {"role": "assistant", "content": "Welcome to the Prism Tiered Interview! Let's build your comprehensive profile."},
                    {"role": "assistant", "content": f"**Tier {agent.current_tier_idx + 1} - {current_q['phase'].title()} Phase**\n\n{current_q['question']}"}
                ]
            else:
                st.session_state.interview_messages = [
                    {"role": "assistant", "content": "✅ Interview complete, but no questions found."}
                ]
        else:
            st.session_state.tiered_interview_agent = None
            st.session_state.interview_messages = [
                {"role": "assistant", "content": "✅ Tiered interview already complete."}
            ]

    agent = st.session_state.get("tiered_interview_agent")
    
    if not agent:
        st.info("✅ Tiered interview already complete—no further questions needed.")
    else:
        # Show current tier progress
        tier_key = agent.get_current_tier_key()
        if tier_key:
            st.info(f"**Current Progress:** Tier {agent.current_tier_idx + 1}/{len(agent.tier_keys)} - {agent.current_phase.title()} Phase")
        
        # Render chat history
        for msg in st.session_state.interview_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Callback to process user reply
    def handle_tiered_reply():
        user_input = st.session_state.get("user_input_tiered", "")
        agent = st.session_state.get("tiered_interview_agent")

        if not agent or not user_input.strip():
            return

        # Append user message
        st.session_state.interview_messages.append({
            "role": "user", "content": user_input
        })

        # Submit to agent
        success = agent.submit_answer(user_input)
        
        if success:
            # Save updates to Firestore
            if agent.save_to_firestore():
                # Check if interview is complete
                if agent.is_complete():
                    st.session_state.interview_messages.append({
                        "role": "assistant",
                        "content": "🎉 **Congratulations!** You have completed the entire tiered interview. Your comprehensive profile has been saved successfully!"
                    })
                    # Clear the agent as interview is complete
                    st.session_state.tiered_interview_agent = None
                else:
                    # Get next question
                    current_q = agent.get_current_question()
                    if current_q:
                        phase_info = f"**Tier {agent.current_tier_idx + 1} - {current_q['phase'].title()} Phase**"
                        
                        # Regenerate question with motivation
                        motivated_question = agent.regenerate_question_with_motivation(current_q['question'])
                        
                        st.session_state.interview_messages.append({
                            "role": "assistant",
                            "content": f"{phase_info}\n\n{motivated_question}"
                        })


                    else:
                        st.session_state.interview_messages.append({
                            "role": "assistant",
                            "content": "⚠️ No more questions available."
                        })
            else:
                st.session_state.interview_messages.append({
                    "role": "assistant",
                    "content": "❌ Failed to save your response. Please try again."
                })
        else:
            st.session_state.interview_messages.append({
                "role": "assistant",
                "content": "❌ Failed to process your answer. Please try again."
            })

    # Render chat input only if agent exists
    if agent:
        st.chat_input(
            "Your answer…",
            key="user_input_tiered",
            on_submit=handle_tiered_reply
        )