
import streamlit as st
import json
import re

import nltk

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="College Helpdesk",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# 2. NLTK SETUP
# ==================================================

tokenizer = TreebankWordTokenizer()


# ==================================================
# 3. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #F7F6F0;
    color: #333B33;
}

header[data-testid="stHeader"] {
    background-color: #F7F6F0;
}

.block-container {
    max-width: 1050px;
    padding-top: 35px;
    padding-bottom: 100px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #E8EDE3;
    border-right: 1px solid #D6DFD1;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #294B3B !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #3D493E !important;
}

/* CLEAR CONVERSATION BUTTON */

section[data-testid="stSidebar"] div.stButton > button {
    background-color: #496B52 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div.stButton > button p,
section[data-testid="stSidebar"] div.stButton > button span {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #294B3B !important;
    color: #FFFFFF !important;
}

/* MAIN HEADING */

.main-title {
    text-align: center;
    color: #294B3B;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 8px;
}

.main-subtitle {
    text-align: center;
    color: #747B70;
    font-size: 16px;
    margin-bottom: 35px;
}

/* WELCOME CARD */

.welcome-card {
    background-color: #E8EDE3;
    border: 1px solid #D6DFD1;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    text-align: center;
}

.welcome-card h3 {
    color: #294B3B;
}

.welcome-card p {
    color: #596458;
    font-size: 15px;
}

/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border: 1px solid #E1E3DA;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"] p {
    color: #333B33 !important;
    line-height: 1.7;
}

/* CHAT INPUT */

[data-testid="stChatInput"] {
    background-color: #F7F6F0 !important;
    padding: 12px 0 !important;
}

[data-testid="stChatInput"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D6DFD1 !important;
    border-radius: 12px !important;
    padding: 5px 12px !important;
    box-shadow: none !important;
    align-items: center !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #333B33 !important;
    padding: 10px 5px !important;
    line-height: 1.5 !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #858B80 !important;
    opacity: 1;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #496B52 !important;
}

/* SUGGESTED QUESTIONS */

div.stButton > button {
    background-color: #FFFFFF;
    color: #355C46 !important;
    border: 1px solid #D6DFD1;
    border-radius: 10px;
    min-height: 48px;
}

div.stButton > button:hover {
    background-color: #E8EDE3;
    border-color: #496B52;
    color: #294B3B !important;
}

/* RESPONSIVE DESIGN */

@media (max-width: 600px) {

    .main-title {
        font-size: 28px;
    }

    .main-subtitle {
        font-size: 14px;
    }

}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD FAQ DATABASE
# ==================================================

@st.cache_data
def load_faqs():

    with open("faqs.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("FAQ database is empty or invalid.")

    for faq in data:

        if "question" not in faq or "answer" not in faq:
            raise ValueError(
                "Each FAQ must contain a question and an answer."
            )

    return data


try:

    faqs = load_faqs()

except Exception as error:

    st.error(f"Unable to load FAQ database: {error}")
    st.stop()


# ==================================================
# 5. FAQ QUESTION VARIATIONS
# ==================================================

faq_variations = {

    "What are the college working hours?": [
        "When does the college open?",
        "When does the college close?",
        "What time does college start?",
        "What time does college end?",
        "Tell me the college timings.",
        "What are the college hours?",
        "When should students come to college?",
        "What is the college schedule?",
        "How many hours is college open?",
        "College opening and closing time"
    ],

    "What are the library timings?": [
        "When does the library open?",
        "When does the library close?",
        "At what time can I visit the library?",
        "What time is the library available?",
        "Tell me about library opening hours.",
        "Is the library open on Saturdays?",
        "What are the library working hours?",
        "When can students use the library?"
    ],

    "How can I apply for admission?": [
        "How do I join this college?",
        "What is the admission process?",
        "How can I get admission?",
        "How do I enroll in college?",
        "Tell me about college admissions.",
        "Where can I apply for college?",
        "How do I get a seat in this college?",
        "What is the procedure to join?"
    ],

    "How can I pay my college fees?": [
        "How do I pay my fees?",
        "What are the fee payment methods?",
        "Where can I pay my college fees?",
        "How can I complete my fee payment?",
        "Can I pay fees online?",
        "Where is the fee payment portal?",
        "How do students pay their fees?",
        "How can I submit my tuition fees?"
    ],

    "When will the examinations be conducted?": [
        "When are the exams?",
        "What is the exam schedule?",
        "When will exams start?",
        "Tell me about examination dates.",
        "When are semester exams conducted?",
        "When is my semester examination?",
        "Where can I check exam dates?",
        "When will the college conduct exams?"
    ],

    "What is the minimum attendance requirement?": [
        "How much attendance is required?",
        "What is the attendance percentage?",
        "How many classes should I attend?",
        "What happens if my attendance is low?",
        "How much attendance should students maintain?",
        "What is the required attendance percentage?",
        "How many days can I be absent?"
    ],

    "Does the college provide transportation?": [
        "Does the college have buses?",
        "Are college bus facilities available?",
        "Is transportation available for students?",
        "Tell me about college bus services.",
        "What bus facilities does the college offer?",
        "Are college buses available?",
        "Does the college provide bus facilities?",
        "Is there a college bus service?",
        "Can I travel to college by bus?",
"Can I go to college by bus?",
"Are buses available for students?",
"How can I travel to college?",
"Does the college have transport facilities?",
"Can I use college transportation?",
    ],

    "How can I contact the placement cell?": [
        "How can I get placement opportunities?",
        "Where is the placement department?",
        "How do I contact placement officers?",
        "Tell me about campus placements.",
        "How can I get a job through college?",
        "Who handles campus placements?",
        "How can I contact the placement team?",
        "Where can I get placement information?"
    ],

    "Are internships available for students?": [
        "Can students apply for internships?",
        "Does the college offer internships?",
        "Where can I find internship opportunities?",
        "How do I get an internship?",
        "Are there internship programs?",
        "Can I do an internship while studying?",
        "How can students find internships?",
        "Does the college support internships?"
    ],

    "What facilities are available in the college?": [
        "What facilities does the college have?",
        "Does the college have laboratories?",
        "Are sports facilities available?",
        "What amenities are provided to students?",
        "Tell me about college facilities.",
        "Does the college have computer labs?",
        "What facilities are provided for students?",
        "What infrastructure does the college have?"
    ],

    "How can I get my ID card?": [
        "Where can I get my student ID?",
        "How do I apply for an ID card?",
        "What should I do if I lose my ID card?",
        "How can I get a new college ID?",
        "Where can I collect my identity card?",
        "How do I get my college identity card?",
        "Who provides student ID cards?"
    ],

    "Is there a canteen in the college?": [
        "Does the college have a canteen?",
        "Where can students get food?",
        "Is a cafeteria available?",
        "Can students buy food on campus?",
        "Is there a food court in college?",
        "Where is the college canteen?",
        "Can I get lunch inside the college?"
    ],

    "How can I get a bonafide certificate?": [
        "Where can I apply for a bonafide?",
        "How do I obtain a bonafide certificate?",
        "Who issues bonafide certificates?",
        "I need a bonafide certificate.",
        "How can I request a bonafide?",
        "Where can I get my bonafide certificate?",
        "How do I apply for a bonafide document?"
    ],

    "Where can I find the academic calendar?": [
        "Where can I see the academic schedule?",
        "How can I get the college calendar?",
        "Where are academic dates published?",
        "Tell me about the academic calendar.",
        "Where can I download the academic calendar?",
        "How do I check the academic schedule?",
        "When does the academic year start?"
    ],

    "Who should I contact for technical support?": [
        "Who can help me with technical problems?",
        "Where can I get IT support?",
        "How do I contact the technical team?",
        "Who should I contact for computer issues?",
        "Who handles technical problems?",
        "Where can I get computer support?",
        "How can I report a technical issue?"
    ]

}


# ==================================================
# 6. TEXT PREPROCESSING
# ==================================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = tokenizer.tokenize(text)

    return " ".join(tokens)


# ==================================================
# 7. PREPARE TRAINING DATA
# ==================================================

training_questions = []
training_answers = []

for faq in faqs:

    original_question = faq["question"]

    variations = faq_variations.get(
        original_question,
        []
    )

    all_questions = [original_question] + variations

    for question in all_questions:

        processed_question = preprocess_text(question)

        if processed_question:

            training_questions.append(processed_question)

            training_answers.append(faq["answer"])


# ==================================================
# 8. WORD-BASED TF-IDF
# ==================================================

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True
)

word_vectors = word_vectorizer.fit_transform(
    training_questions
)


# ==================================================
# 9. CHARACTER-BASED TF-IDF
# ==================================================

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    sublinear_tf=True
)

char_vectors = char_vectorizer.fit_transform(
    training_questions
)


# ==================================================
# 10. FIND BEST ANSWER
# ==================================================

# College-topic words prevent unrelated questions from being forced
# into the closest FAQ match.
COLLEGE_KEYWORDS = {
    "college", "campus", "student", "students",
    "library", "book", "books",
    "admission", "admissions", "apply", "enroll", "enrollment",
    "fee", "fees", "payment", "payments", "tuition", "accounts",
    "exam", "exams", "examination", "examinations", "semester",
    "attendance", "absent", "absence", "classes",
    "bus", "buses", "transport", "transportation",
    "placement", "placements", "recruitment", "internship", "internships",
    "facility", "facilities", "laboratory", "laboratories", "lab", "labs",
    "sports", "infrastructure", "canteen", "cafeteria", "food",
    "id", "card", "identity",
    "bonafide", "certificate", "academic", "calendar", "schedule",
    "technical", "technology", "computer", "support"
}

COLLEGE_PHRASES = (
    "college", "campus", "student", "library", "admission",
    "fee", "fees", "payment", "exam", "examination", "semester",
    "attendance", "bus", "transport", "placement", "internship",
    "facility", "facilities", "canteen", "cafeteria", "id card",
    "identity card", "bonafide", "academic calendar",
    "technical support", "it support"
)


def is_college_related(question):
    normalized = preprocess_text(question)
    words = set(normalized.split())

    if words.intersection(COLLEGE_KEYWORDS):
        return True

    lowered = question.lower()
    return any(phrase in lowered for phrase in COLLEGE_PHRASES)


def get_answer(user_question):

    processed_input = preprocess_text(user_question)

    if not processed_input:
        return "Please enter a meaningful question.", 0.0

    # Reject out-of-scope questions before searching the FAQ database.
    if not is_college_related(user_question):
        return (
            "I'm sorry, I can only answer questions related to "
            "college information, such as admissions, fees, library "
            "timings, examinations, transportation and placements. "
            "Please ask me a college-related question.",
            0.0
        )

    user_word_vector = word_vectorizer.transform([processed_input])
    user_char_vector = char_vectorizer.transform([processed_input])

    word_scores = cosine_similarity(
        user_word_vector,
        word_vectors
    )[0]

    char_scores = cosine_similarity(
        user_char_vector,
        char_vectors
    )[0]

    # Word matches matter more than character-level resemblance.
    combined_scores = (0.90 * word_scores) + (0.10 * char_scores)

    best_match_index = int(combined_scores.argmax())
    best_score = float(combined_scores[best_match_index])
    best_word_score = float(word_scores[best_match_index])

    # Require both a reasonable combined score and a word-level match.
    if best_score < 0.28 or best_word_score < 0.20:
        return (
            "Sorry, I couldn't find a sufficiently relevant answer "
            "in my college FAQ database. Please try asking about "
            "admissions, fees, library timings, examinations, "
            "attendance, transportation or placements.",
            best_score
        )

    answer = training_answers[best_match_index]
    return answer, best_score


# ==================================================
# 11. SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = ""


# ==================================================
# 12. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🎓 College Helpdesk")

    st.write(
        "Your virtual assistant for college-related information."
    )

    st.divider()

    st.markdown("### What I can help with")

    st.markdown("""
    📚 Library Information

    🎓 Admissions

    💳 Fee Payments

    📝 Examinations

    🚌 Transportation

    💼 Placements

    📅 College Timings
    """)

    st.divider()

    st.markdown("### Built using")

    st.markdown("""
    - Python
    - Streamlit
    - NLTK
    - TF-IDF
    - Cosine Similarity
    """)

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = ""

        st.rerun()


# ==================================================
# 13. MAIN HEADING
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🎓 College Helpdesk Chatbot'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Your friendly assistant for college-related questions.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# 14. WELCOME SCREEN
# ==================================================

if not st.session_state.messages:

    with st.container(border=True):

        st.markdown(
            "<h3 style='text-align:center; color:#294B3B;'>"
            "Hello! Welcome to the College Helpdesk 👋"
            "</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center; color:#596458;'>"
            "Have a question about your college? "
            "I'm here to help you find the information you need."
            "</p>",
            unsafe_allow_html=True
        )

    st.markdown("### 💡 Try asking")

    suggestions = [
        "What are the college working hours?",
        "What are the library timings?",
        "How can I pay my college fees?",
        "Does the college provide transportation?"
    ]

    col1, col2 = st.columns(2)

    for index, question in enumerate(suggestions):

        with (col1 if index % 2 == 0 else col2):

            if st.button(
                question,
                key=f"suggestion_{index}",
                use_container_width=True
            ):

                st.session_state.pending_question = question

                st.rerun()




# ==================================================
# 15. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==================================================
# 16. CHAT INPUT
# ==================================================

user_question = st.chat_input(
    "Ask me anything about your college..."
)


# ==================================================
# 17. HANDLE SUGGESTED QUESTIONS
# ==================================================

if st.session_state.pending_question:

    user_question = st.session_state.pending_question

    st.session_state.pending_question = ""


# ==================================================
# 18. PROCESS USER QUESTION
# ==================================================

if user_question:

    # Save user question

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Generate FAQ answer

    answer, score = get_answer(user_question)

    # Save chatbot answer

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Refresh the application

    st.rerun()
