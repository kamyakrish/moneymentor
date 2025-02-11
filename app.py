import streamlit as st
import os
import faiss
import openai
from openai import OpenAI
import numpy as np
import pandas as pd

# Ensure environment variables are loaded first
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.idx")
metadata = pd.read_pickle("faiss_metadata.pkl")

# Store conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def extract_prioritized_terms():
    return set(metadata["term"].str.lower().str.strip())

prioritized_terms = extract_prioritized_terms()

def search_faiss_or_gpt(query, top_k=5):
    """Search FAISS for the closest financial term. If no match, use OpenAI."""
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
    
    D, I = index.search(query_embedding, top_k)
    query_lower = query.lower().strip()
    query_words = set(query_lower.replace("?", "").split())
    
    best_match_term, best_match_definition = None, None
    valid_matches = []

    for i in range(top_k):
        idx = I[0][i]
        if idx < len(metadata):
            term = metadata.iloc[idx]["term"]
            definition = metadata.iloc[idx]["definition"]
            term_lower = term.lower()
            term_words = set(term_lower.split())
            if query_words & term_words:
                valid_matches.append((term, definition))
    
    if valid_matches:
        best_match_term, best_match_definition = valid_matches[0]
        return best_match_term, best_match_definition, False
    
    term, definition = query, generate_definition_openai(query)
    return term, definition, True

def generate_definition_openai(query):
    """Ask OpenAI to generate a definition when FAISS has no match."""
    messages = [
        {"role": "system", "content": "You are a financial expert. Provide clear and accurate definitions."},
        {"role": "user", "content": f"Define '{query}' as a financial term concisely."}
    ]
    ai_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return ai_response.choices[0].message.content.strip()

def concise_definition_agent(term, definition):
    """Generate a concise financial definition."""
    messages = [
        {"role": "system", "content": "You are a financial assistant."},
        {"role": "user", "content": f"Provide a concise definition for '{term}':\n{definition}"}
    ]
    ai_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return ai_response.choices[0].message.content.strip()

def simplified_explanation_agent(term, definition):
    """Provide a simplified explanation or analogy."""
    messages = [
        {"role": "system", "content": "You are a friendly financial educator."},
        {"role": "user", "content": f"Explain '{term}' in simple terms, using an analogy if possible."}
    ]
    ai_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return ai_response.choices[0].message.content.strip()

def source_recommendation_agent(term, definition):
    """Recommend reputable sources for further reading."""
    messages = [
        {"role": "system", "content": "You recommend reputable financial sources."},
        {"role": "user", "content": f"Recommend reliable sources for learning about '{term}'."}
    ]
    ai_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return ai_response.choices[0].message.content.strip()

def follow_up_question_agent(term, definition):
    """Generate a follow-up question to keep the conversation going."""
    messages = [
        {"role": "system", "content": "You are a financial education assistant. Generate a simple to understand question about to encourage th user to keep learning"},
        {"role": "user", "content": f"Given the financial term '{term}' and its definition: {definition}, what is a good follow-up question that encourages deeper understanding?"}
    ]
    ai_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return ai_response.choices[0].message.content.strip()

def generate_ai_response(user_query, term, definition):
    concise_def = concise_definition_agent(term, definition)
    simplified_exp = simplified_explanation_agent(term, definition)
    source_rec = source_recommendation_agent(term, definition)
    follow_up_q = follow_up_question_agent(term, definition)
    
    return (
        f"💡 **{term}**\n\n"
        f"📘 **Definition:** {concise_def}\n\n"
        f"🔍 **Explanation:** {simplified_exp}\n\n"
        f"📚 **Learn More:** {source_rec}\n\n"
        f"🤔 **Follow-Up Question:** {follow_up_q}\n"
    )

    

st.set_page_config(page_title="💬 Money Mentor", layout="centered")

# Custom CSS for styling and seamless infinite ticker animation
st.markdown(
    """
    <style>
    .main {
        background-color: white;
        display: flex;
        flex-direction: column;
        justify-content: flex-start; /* Align content to the top */
        min-height: 100vh;
        padding: 0 20px; /* Minimal padding for alignment */
    }
    .stButton > button {
        color: white;
        background: linear-gradient(90deg, #4a90e2, #007aff);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    input {
        border: 1px solid #e2e8f0;
        padding: 0.75rem;
        border-radius: 8px;
    }
    h1 {
        color: black;
        font-weight: bold;
        margin: 10px 0; /* Minimal margin for alignment */
    }
    h2, h3 {
        color: black;
        margin: 5px 0; /* Minimal margin for alignment */
    }

    /* Spacing between subtitle and input field */
    .subheader {
        margin-bottom: 20px !important;
    }

    /* Ticker Styles */
    .ticker-wrapper {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        position: relative;
        background: #fff;
        padding: 8px 0;
        margin-bottom: 10px; /* Slight spacing between tickers */
    }

    .ticker {
        display: inline-block;
        min-width: 200%; /* Ensures the content repeats seamlessly */
    }

    /* Animation for Ticker 1 (Left to Right) */
    .ticker-1 {
        animation: ticker-left 45s linear infinite;
        color: #3c19a2;
        font-weight: bold; /* Bold text */
    }

    /* Animation for Ticker 2 (Right to Left) */
    .ticker-2 {
        animation: ticker-right 45s linear infinite;
        color: #820b5c;
        font-weight: bold; /* Bold text */
    }

    .ticker span {
        font-size: 16px;
        padding: 0 20px;
        font-family: 'Calibri', sans-serif;
    }

    /* Keyframes for Left-to-Right Ticker */
    @keyframes ticker-left {
        from { transform: translateX(-50%); }
        to { transform: translateX(0%); }
    }

    /* Keyframes for Right-to-Left Ticker */
    @keyframes ticker-right {
        from { transform: translateX(0%); }
        to { transform: translateX(-50%); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render two tickers with duplicated content for seamless scrolling
st.markdown(
    """
    <div class="ticker-wrapper">
        <div class="ticker ticker-1">
            <span>Investing Strategies</span>
            <span>Retirement Planning</span>
            <span>Compound Interest</span>
            <span>Stock Market</span>
            <span>Wealth Management</span>
            <span>Cryptocurrency Basics</span>
            <span>Index Funds</span>
            <span>Mutual Funds</span>
            <span>Budgeting Tips</span>
            <span>Emergency Fund</span>
            <span>Financial Literacy</span>
            <span>Credit Card Management</span>
            <span>Inflation Protection</span>
            <span>Tax Efficiency</span>
            <span>Debt Reduction</span>
            <span>Risk Assessment</span>
            <span>Portfolio Diversification</span>
            <span>Real Estate Investments</span>
            <span>Expense Tracking</span>
            <span>Retirement Accounts</span>
            <span>Dividend Stocks</span>
            <span>Insurance Planning</span>
            <span>Long-Term Savings</span>
            <span>401(k) Management</span>
            <span>Asset Allocation</span>
        </div>
    </div>
    <div class="ticker-wrapper">
        <div class="ticker ticker-2">
            <span>Wealth Building</span>
            <span>Tax Planning</span>
            <span>Expense Optimization</span>
            <span>Financial Goals</span>
            <span>Capital Gains</span>
            <span>Estate Planning</span>
            <span>Hedge Funds</span>
            <span>Social Security</span>
            <span>Financial Independence</span>
            <span>Investment Banking</span>
            <span>Corporate Bonds</span>
            <span>Day Trading</span>
            <span>Passive Income</span>
            <span>Cash Flow Management</span>
            <span>Health Savings Accounts</span>
            <span>Economic Indicators</span>
            <span>Stock Options</span>
            <span>Interest Rates</span>
            <span>Financial Planning</span>
            <span>Monetary Policy</span>
            <span>Angel Investing</span>
            <span>Private Equity</span>
            <span>Venture Capital</span>
            <span>Startup Funding</span>
            <span>Financial Risk</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Page title
st.title("💬 Money Mentor")

# Subtitle with extra spacing
st.subheader("Ask me about financial terms, and I'll provide definitions, explanations, and sources!")
st.markdown("<div class='subheader'></div>", unsafe_allow_html=True)  # Adds spacing below the subtitle


user_input = st.text_input("📝 Your question:")
if user_input:
    if user_input.lower() in ["exit", "quit"]:
        st.write("👋 Goodbye!")
    else:
        term, definition, is_ai_generated = search_faiss_or_gpt(user_input)
        response = generate_ai_response(user_input, term, definition)
        st.session_state.conversation.append({"user": user_input, "bot": response})
        st.write(response)

# Display Conversation History
st.write("### 🗨️ Conversation History")
for entry in st.session_state.conversation:
    st.write(f"**You:** {entry['user']}")
    st.write(f"**Bot:** {entry['bot']}")
    st.write("---")
