import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
import plotly.express as px
from fpdf import FPDF

# -----------------------
# CONFIG
# -----------------------

BG_PATH = "C:/Users/Deepika T/OneDrive/Desktop/finger print photos/background.jpg"

DEFAULT_CATEGORIES = [
    "Groceries","Rent","Utilities","Transport",
    "Dining","Entertainment","Salary","Health","Shopping","Other"
]

# -----------------------
# PAGE CONFIG & STYLE
# -----------------------
st.set_page_config(page_title="Budget Buddy", page_icon="💸", layout="wide")

def add_bg_local(image_path):
    if not os.path.exists(image_path):
        st.warning(f"Background image not found at: {image_path}")
        return
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .glass {{
            background: rgba(255,255,255,0.94);
            padding: 12px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_local(BG_PATH)
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.title("💸 Budget Buddy — Personal Finance Assistant")
st.write("Add transactions, generate budgets, and chat with your AI assistant.")

# -----------------------
# SESSION STORAGE
# -----------------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

# -----------------------
# Simple Auth (sidebar)
# -----------------------
with st.sidebar:
    st.header("Account")
    if st.session_state.user is None:
        mode = st.radio("Mode", ["Demo", "Login"])
        if mode == "Demo":
            if st.button("Enter demo user"):
                st.session_state.user = "demo_user"
                st.success("Demo user active")
        else:
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.button("Login"):
                if user == "admin" and pwd == "1234":
                    st.session_state.user = user
                    st.success("Logged in")
                else:
                    st.error("Invalid credentials (use admin / 1234 for demo)")
    else:
        st.markdown(f"Signed in as *{st.session_state.user}*")
        if st.button("Logout"):
            st.session_state.user = None
            st.success("Logged out")

# -----------------------
# IBM Granite Local Model Chat
# -----------------------
@st.cache_resource
def load_model():
    """Load the IBM Granite model and tokenizer"""
    try:
        with st.spinner("Loading AI model... This may take a few minutes on first run."):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "ibm-granite/granite-3.3-2b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            
            st.success("AI model loaded successfully!")
            return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def granite_chat(question):
    """Chat using local IBM Granite model"""
    try:
        if st.session_state.tokenizer is None or st.session_state.model is None:
            return "AI model not loaded. Please refresh the page."
        
        # Prepare the chat message
        messages = [
            {"role": "user", "content": question}
        ]
        
        # Apply chat template and tokenize
        inputs = st.session_state.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        device = next(st.session_state.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with st.spinner("AI is thinking..."):
            outputs = st.session_state.model.generate(
                **inputs, 
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = st.session_state.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return response.strip() if response.strip() else "I couldn't generate a response. Please try again."
        
    except Exception as e:
        return f"Error in AI chat: {str(e)}"

# -----------------------
# Simple category heuristic
# -----------------------
def simple_category(text):
    t = (text or "").lower()
    if any(k in t for k in ["grocery","supermarket","mart","grocer"]): return "Groceries"
    if any(k in t for k in ["rent","landlord","apartment"]): return "Rent"
    if any(k in t for k in ["electric","electricity","water","utility","gas"]): return "Utilities"
    if any(k in t for k in ["uber","taxi","ola","bus","train"]): return "Transport"
    if any(k in t for k in ["restaurant","cafe","coffee","dine"]): return "Dining"
    if any(k in t for k in ["movie","netflix","spotify","subscription"]): return "Entertainment"
    if any(k in t for k in ["salary","pay"]): return "Salary"
    return "Other"

# -----------------------
# Budget Generation
# -----------------------
def generate_budget_plan(transactions):
    if not transactions:
        return {"message":"Add transactions first."}
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df['abs_amount'] = df['amount'].abs()
    three_months_ago = pd.Timestamp.now() - pd.DateOffset(months=3)
    df_recent = df[df['date'] >= three_months_ago]
    if df_recent.empty:
        df_recent = df
    cat_avg = df_recent.groupby('category')['abs_amount'].mean().to_dict()
    plan = {cat: round(cat_avg.get(cat,0)*1.10,2) for cat in sorted(set(df['category'].tolist() or DEFAULT_CATEGORIES))}
    total = round(sum(plan.values()),2)
    return {"plan": plan, "total": total}

# -----------------------
# PDF Export
# -----------------------
def export_budget_pdf(plan):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Budget Buddy - Monthly Budget", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)

    for cat, amt in plan['plan'].items():
        pdf.cell(0, 8, f"{cat}: ₹{amt:.2f}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, f"Total Proposed Budget: ₹{plan['total']:.2f}", ln=True)
    
    filename = f"Budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# -----------------------
# UI: Transactions
# -----------------------
col1, col2 = st.columns([2,1])

with col1:
    st.header("Add Transaction Manually")
    with st.form("manual_tx"):
        d = st.date_input("Date", value=datetime.today())
        desc = st.text_input("Description")

        col_in, col_ex = st.columns(2)
        with col_in:
            income = st.number_input("Income (₹)", min_value=0.0, value=0.0, format="%.2f")
        with col_ex:
            expense = st.number_input("Expense (₹)", min_value=0.0, value=0.0, format="%.2f")

        cat = st.selectbox("Category", DEFAULT_CATEGORIES)

        if st.form_submit_button("Add Transaction"):
            added = False
            if income != 0:
                st.session_state.transactions.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "description": desc,
                    "amount": float(income),
                    "category": cat,
                    "source": "manual"
                })
                added = True
            if expense != 0:
                st.session_state.transactions.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "description": desc,
                    "amount": -abs(float(expense)),
                    "category": cat,
                    "source": "manual"
                })
                added = True
            if added:
                st.success("Transaction added.")
            else:
                st.warning("Please enter a non-zero income or expense.")

    st.markdown("---")
    st.subheader("Transactions (recent)")
    if st.session_state.transactions:
        df_tx = pd.DataFrame(st.session_state.transactions).sort_values("date", ascending=False)
        st.dataframe(df_tx)
    else:
        st.info("No transactions yet. Add manually.")

with col2:
    st.header("Dashboard & Budget")
    if st.session_state.transactions:
        df_all = pd.DataFrame(st.session_state.transactions)
        df_all['date'] = pd.to_datetime(df_all['date'])
        total_income = df_all[df_all['amount']>0]['amount'].sum()
        total_expense = df_all[df_all['amount']<0]['amount'].sum()
        st.metric("Total Income", f"₹{total_income:.2f}")
        st.metric("Total Expenses", f"₹{abs(total_expense):.2f}")
        st.metric("Net", f"₹{(total_income+total_expense):.2f}")

        st.markdown("Spending by Category")
        by_cat = df_all.groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        if not by_cat.empty:
            fig = px.bar(x=by_cat.values, y=by_cat.index, orientation='h', labels={'x':'Amount','y':'Category'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for category chart.")

        st.markdown("Auto Budget Plan")
        plan = generate_budget_plan(st.session_state.transactions)
        if "plan" in plan:
            plan_df = pd.DataFrame(list(plan['plan'].items()), columns=["Category","Proposed Monthly Budget"])
            st.dataframe(plan_df)
            st.write(f"Total proposed monthly budget: ₹{plan['total']:.2f}")

            if st.button("Export Budget as PDF"):
                pdf_file = export_budget_pdf(plan)
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF", f, file_name=pdf_file, mime="application/pdf")
        else:
            st.info(plan.get("message"))
    else:
        st.info("No transactions yet. Add some to see dashboard & budget.")

# -----------------------
# AI Model Loading Section
# -----------------------
st.markdown("---")
st.header("🤖 AI Assistant Setup")

if not st.session_state.model_loaded:
    if st.button("🚀 Load AI Model (IBM Granite)"):
        st.session_state.tokenizer, st.session_state.model = load_model()
        if st.session_state.tokenizer and st.session_state.model:
            st.session_state.model_loaded = True
            st.success("AI model loaded successfully! You can now use the chat feature.")
        else:
            st.error("Failed to load AI model. Please check your internet connection and try again.")
else:
    st.success("✅ AI model is loaded and ready!")
    if st.button("🔄 Reload Model"):
        st.session_state.model_loaded = False
        st.session_state.tokenizer = None
        st.session_state.model = None
        st.rerun()

# -----------------------
# Chatbot
# -----------------------
st.markdown("---")
st.header("Chat with Budget JINI")

if st.session_state.model_loaded:
    question = st.text_input("Ask a question (e.g., 'How much did I spend on groceries?')", key="chat_input")
    if st.button("Send", key="send"):
        if st.session_state.user is None:
            st.error("Please login (or enter demo) to use personalized chat.")
        else:
            resp = granite_chat(question)
            if resp and not resp.startswith("Error"):
                st.success(resp)
            else:
                st.error(resp)
else:
    st.info("Please load the AI model first to use the chat feature.")

st.markdown("</div>", unsafe_allow_html=True)
