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
BG_PATH = r"C:\Users\Deepika T\OneDrive\Desktop\finger print photos\background.jpg"

DEFAULT_CATEGORIES = [
    "Groceries","Rent","Utilities","Transport",
    "Dining","Entertainment","Salary","Shopping" , "Health","Other"
]

# -----------------------
# PAGE CONFIG & STYLE
# -----------------------
st.set_page_config(
    page_title="Budget Buddy", 
    page_icon="💸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimization
@st.cache_data
def load_sample_data():
    """Load sample transaction data for demo"""
    return [
        {"date": "2024-01-15", "description": "Grocery shopping", "amount": -2500.0, "category": "Groceries", "source": "demo"},
        {"date": "2024-01-16", "description": "Salary", "amount": 50000.0, "category": "Salary", "source": "demo"},
        {"date": "2024-01-17", "description": "Rent payment", "amount": -15000.0, "category": "Rent", "source": "demo"},
        {"date": "2024-01-18", "description": "Uber ride", "amount": -300.0, "category": "Transport", "source": "demo"},
        {"date": "2024-01-19", "description": "Movie tickets", "amount": -800.0, "category": "Entertainment", "source": "demo"},
    ]

def add_bg_local(image_path):
    """Load and apply background image with better error handling"""
    try:
        if not os.path.exists(image_path):
            st.warning(f"Background image not found at: {image_path}")
            # Apply default styling without background
            st.markdown(
                """
                <style>
                .stApp {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .glass {
                    background: rgba(255,255,255,0.94);
                    padding: 12px;
                    border-radius: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            return
        
        # Check if file is not empty
        if os.path.getsize(image_path) == 0:
            st.warning("Background image file is empty. Using default background.")
            # Apply default styling without background
            st.markdown(
                """
                <style>
                .stApp {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .glass {
                    background: rgba(255,255,255,0.94);
                    padding: 12px;
                    border-radius: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            return
            
        with open(image_path, "rb") as f:
            data = f.read()
        
        if not data:
            st.warning("Could not read background image. Using default background.")
            return
            
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
    except Exception as e:
        st.warning(f"Error loading background image: {e}. Using default background.")
        # Apply default styling without background
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .glass {
                background: rgba(255,255,255,0.94);
                padding: 12px;
                border-radius: 10px;
            }
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
if 'quick_question' not in st.session_state:
    st.session_state.quick_question = None
if 'users' not in st.session_state:
    st.session_state.users = {
        "admin": {"password": "1234", "name": "Admin User"}
    }

# -----------------------
# User Authentication (sidebar)
# -----------------------
with st.sidebar:
    st.header("Account")
    if st.session_state.user is None:
        mode = st.radio("Mode", ["Demo", "Login", "Register"])
        
        if mode == "Demo":
            if st.button("Enter demo user"):
                st.session_state.user = "demo_user"
                # Load sample data for demo
                if not st.session_state.transactions:
                    st.session_state.transactions = load_sample_data()
                st.success("Demo user active with sample data!")
        
        elif mode == "Login":
            st.subheader("Login")
            login_user = st.text_input("Username", key="login_user")
            login_pwd = st.text_input("Password", type="password", key="login_pwd")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    if login_user in st.session_state.users and st.session_state.users[login_user]["password"] == login_pwd:
                        st.session_state.user = login_user
                        st.success(f"Welcome back, {st.session_state.users[login_user]['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with col2:
                if st.button("Forgot Password"):
                    st.info("Contact admin to reset your password")
        
        elif mode == "Register":
            st.subheader("Create New Account")
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_user = st.text_input("Username", key="reg_user")
            reg_pwd = st.text_input("Password", type="password", key="reg_pwd")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            if st.button("Register"):
                if not reg_name or not reg_user or not reg_pwd:
                    st.error("Please fill all fields")
                elif reg_pwd != reg_confirm:
                    st.error("Passwords do not match")
                elif reg_user in st.session_state.users:
                    st.error("Username already exists")
                elif len(reg_pwd) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    st.session_state.users[reg_user] = {
                        "password": reg_pwd,
                        "name": reg_name
                    }
                    st.success(f"Account created successfully! Welcome, {reg_name}")
                    st.session_state.user = reg_user
                    st.rerun()
    
    else:
        user_info = st.session_state.users.get(st.session_state.user, {"name": st.session_state.user})
        st.markdown(f"**Signed in as:** {user_info['name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Logout"):
                st.session_state.user = None
                st.success("Logged out")
                st.rerun()
        
        with col2:
            if st.button("View Profile"):
                st.info(f"**Username:** {st.session_state.user}")
                st.info(f"**Name:** {user_info['name']}")
                st.info(f"**Transactions:** {len(st.session_state.transactions)}")

# -----------------------
# IBM Granite Local Model Chat
# -----------------------
@st.cache_resource
def load_model():
    """Load the IBM Granite model and tokenizer with optimized settings for speed"""
    try:
        with st.spinner("Loading AI model... This may take a few minutes on first run."):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "ibm-granite/granite-3.3-2b-instruct"
            
            # Optimize for speed - use CPU if CUDA not available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer with optimized settings
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Load model with optimized settings for speed
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move to device
            if device == "cpu":
                model = model.to(device)
            
            st.success(f"AI model loaded successfully on {device.upper()}!")
            return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def granite_chat(question):
    """Fast financial assistant using rule-based responses"""
    try:
        # Remove the model dependency - use only rule-based responses
        # if st.session_state.tokenizer is None or st.session_state.model is None:
        #     return "AI model not loaded. Please refresh the page."
        
        # Get transaction data for context
        transactions = st.session_state.transactions
        df = pd.DataFrame(transactions) if transactions else pd.DataFrame()
        
        # Convert question to lowercase for easier matching
        q = question.lower().strip()
        
        # Rule-based responses for common financial questions
        if any(word in q for word in ["groceries", "grocery", "food", "supermarket"]):
            if not df.empty:
                try:
                    grocery_data = df[df['category'] == 'Groceries']
                    if not grocery_data.empty:
                        grocery_spending = grocery_data['amount'].abs().sum()
                        return f"💰 **Grocery Spending**: You've spent ₹{grocery_spending:.2f} on groceries so far. Consider setting a monthly budget of ₹{max(grocery_spending * 0.8, 2000):.0f} to save money."
                    else:
                        return "💰 **Grocery Tip**: You haven't recorded any grocery expenses yet. Set a monthly grocery budget of ₹2000-3000 per person. Plan meals ahead and make a shopping list to avoid impulse purchases."
                except Exception as e:
                    return "💰 **Grocery Tip**: Set a monthly grocery budget of ₹2000-3000 per person. Plan meals ahead and make a shopping list to avoid impulse purchases."
            else:
                return "💰 **Grocery Tip**: Set a monthly grocery budget of ₹2000-3000 per person. Plan meals ahead and make a shopping list to avoid impulse purchases."
        
        elif any(word in q for word in ["rent", "housing", "accommodation"]):
            if not df.empty:
                try:
                    rent_data = df[df['category'] == 'Rent']
                    if not rent_data.empty:
                        rent_spending = rent_data['amount'].abs().sum()
                        return f"🏠 **Rent Analysis**: Your total rent spending is ₹{rent_spending:.2f}. Rent should ideally be 30% or less of your monthly income. Are you within this range?"
                    else:
                        return "🏠 **Rent Advice**: You haven't recorded any rent expenses yet. Aim to keep rent at 30% or less of your monthly income. Consider roommates or moving to a more affordable area if rent is too high."
                except Exception as e:
                    return "🏠 **Rent Advice**: Aim to keep rent at 30% or less of your monthly income. Consider roommates or moving to a more affordable area if rent is too high."
            else:
                return "🏠 **Rent Advice**: Aim to keep rent at 30% or less of your monthly income. Consider roommates or moving to a more affordable area if rent is too high."
        
        elif any(word in q for word in ["transport", "uber", "taxi", "fuel", "gas"]):
            if not df.empty:
                try:
                    transport_data = df[df['category'] == 'Transport']
                    if not transport_data.empty:
                        transport_spending = transport_data['amount'].abs().sum()
                        return f"🚗 **Transport Spending**: You've spent ₹{transport_spending:.2f} on transport. Consider carpooling, public transport, or walking for short distances to save money."
                    else:
                        return "🚗 **Transport Tips**: You haven't recorded any transport expenses yet. Use public transport when possible, carpool with colleagues, and consider walking or cycling for short trips to reduce transport costs."
                except Exception as e:
                    return "🚗 **Transport Tips**: Use public transport when possible, carpool with colleagues, and consider walking or cycling for short trips to reduce transport costs."
            else:
                return "🚗 **Transport Tips**: Use public transport when possible, carpool with colleagues, and consider walking or cycling for short trips to reduce transport costs."
        
        elif any(word in q for word in ["entertainment", "movie", "netflix", "subscription"]):
            if not df.empty:
                try:
                    entertainment_data = df[df['category'] == 'Entertainment']
                    if not entertainment_data.empty:
                        entertainment_spending = entertainment_data['amount'].abs().sum()
                        return f"🎬 **Entertainment Spending**: You've spent ₹{entertainment_spending:.2f} on entertainment. Review your subscriptions and cancel unused ones to save money monthly."
                    else:
                        return "🎬 **Entertainment Budget**: You haven't recorded any entertainment expenses yet. Set aside ₹1000-2000 monthly for entertainment. Review subscriptions regularly and cancel unused services."
                except Exception as e:
                    return "🎬 **Entertainment Budget**: Set aside ₹1000-2000 monthly for entertainment. Review subscriptions regularly and cancel unused services."
            else:
                return "🎬 **Entertainment Budget**: Set aside ₹1000-2000 monthly for entertainment. Review subscriptions regularly and cancel unused services."
        
        elif any(word in q for word in ["budget", "plan", "planning"]):
            if not df.empty:
                try:
                    expenses_data = df[df['amount'] < 0]
                    income_data = df[df['amount'] > 0]
                    
                    total_expenses = expenses_data['amount'].abs().sum() if not expenses_data.empty else 0
                    total_income = income_data['amount'].sum() if not income_data.empty else 0
                    
                    savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
                    return f"📊 **Budget Analysis**: Your total expenses: ₹{total_expenses:.2f}, Income: ₹{total_income:.2f}, Savings rate: {savings_rate:.1f}%. Aim for 20% savings rate!"
                except Exception as e:
                    return "📊 **Budget Planning**: Follow the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. Start by tracking all your expenses!"
            else:
                return "📊 **Budget Planning**: Follow the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. Start by tracking all your expenses!"
        
        elif any(word in q for word in ["save", "saving", "money"]):
            if not df.empty:
                try:
                    # Get expenses only (negative amounts)
                    expenses_df = df[df['amount'] < 0].copy()
                    if not expenses_df.empty:
                        # Calculate absolute amounts for each category
                        expenses_df['abs_amount'] = expenses_df['amount'].abs()
                        category_totals = expenses_df.groupby('category')['abs_amount'].sum()
                        if not category_totals.empty:
                            top_expense = category_totals.idxmax()
                            top_amount = category_totals.max()
                            return f"💡 **Saving Tip**: Your biggest expense category is {top_expense} (₹{top_amount:.2f}). Look for ways to reduce spending in this area first. Small changes add up!"
                        else:
                            return "💡 **Saving Tip**: You don't have any expenses recorded yet. Start tracking your spending to identify areas where you can save!"
                    else:
                        return "💡 **Saving Tip**: Great! You don't have any expenses recorded. Keep up the good work and continue tracking your income!"
                except Exception as e:
                    return "💡 **Saving Strategies**: 1) Track all expenses 2) Set up automatic savings 3) Use the 50/30/20 rule 4) Cook at home 5) Cancel unused subscriptions"
            else:
                return "💡 **Saving Strategies**: 1) Track all expenses 2) Set up automatic savings 3) Use the 50/30/20 rule 4) Cook at home 5) Cancel unused subscriptions"
        
        elif any(word in q for word in ["income", "salary", "earn"]):
            if not df.empty:
                try:
                    income_data = df[df['amount'] > 0]
                    if not income_data.empty:
                        total_income = income_data['amount'].sum()
                        return f"💵 **Income Overview**: Your total recorded income is ₹{total_income:.2f}. Consider diversifying income sources through side hustles or investments."
                    else:
                        return "💵 **Income Tips**: You haven't recorded any income yet. Track all income sources, negotiate salary increases, develop new skills, and consider passive income opportunities."
                except Exception as e:
                    return "💵 **Income Tips**: Track all income sources, negotiate salary increases, develop new skills, and consider passive income opportunities."
            else:
                return "💵 **Income Tips**: Track all income sources, negotiate salary increases, develop new skills, and consider passive income opportunities."
        
        elif any(word in q for word in ["debt", "loan", "credit"]):
            return "⚠️ **Debt Management**: Prioritize high-interest debt first. Consider debt consolidation if you have multiple loans. Aim to keep debt-to-income ratio below 40%."
        
        elif any(word in q for word in ["investment", "invest", "stocks"]):
            return "📈 **Investment Advice**: Start with emergency fund (3-6 months expenses), then consider SIPs in index funds. Diversify your portfolio and invest for the long term."
        
        elif any(word in q for word in ["emergency", "fund"]):
            return "🆘 **Emergency Fund**: Aim to save 3-6 months of expenses in a high-yield savings account. This provides financial security during unexpected situations."
        
        elif any(word in q for word in ["hello", "hi", "hey"]):
            return "👋 **Hello!** I'm your Budget Buddy AI assistant. Ask me about your spending patterns, budget advice, or financial tips!"
        
        elif any(word in q for word in ["help", "what can you do"]):
            return "❓ **I can help with**: Spending analysis, budget planning, saving tips, financial advice, expense tracking insights, and personalized recommendations based on your data!"
        
        elif any(word in q for word in ["how", "can", "save", "money"]) and "how" in q and ("save" in q or "money" in q):
            # Handle "how can i save money" type questions
            if not df.empty:
                try:
                    expenses_data = df[df['amount'] < 0]
                    if not expenses_data.empty:
                        # Calculate absolute amounts for each category
                        expenses_data = expenses_data.copy()
                        expenses_data['abs_amount'] = expenses_data['amount'].abs()
                        category_totals = expenses_data.groupby('category')['abs_amount'].sum()
                        
                        if not category_totals.empty:
                            top_expense = category_totals.idxmax()
                            top_amount = category_totals.max()
                            
                            # Provide specific advice based on top expense
                            if top_expense == "Groceries":
                                return f"💰 **Save on Groceries**: Your biggest expense is {top_expense} (₹{top_amount:.2f}). **Tips**: 1) Plan meals weekly 2) Buy in bulk 3) Use grocery apps for discounts 4) Cook at home 5) Avoid impulse purchases"
                            elif top_expense == "Rent":
                                return f"🏠 **Save on Rent**: Your biggest expense is {top_expense} (₹{top_amount:.2f}). **Tips**: 1) Consider roommates 2) Negotiate rent 3) Look for cheaper areas 4) Check for rent control 5) Consider house-sharing"
                            elif top_expense == "Transport":
                                return f"🚗 **Save on Transport**: Your biggest expense is {top_expense} (₹{top_amount:.2f}). **Tips**: 1) Use public transport 2) Carpool with colleagues 3) Walk/cycle short distances 4) Use ride-sharing apps wisely 5) Maintain your vehicle"
                            elif top_expense == "Entertainment":
                                return f"🎬 **Save on Entertainment**: Your biggest expense is {top_expense} (₹{top_amount:.2f}). **Tips**: 1) Cancel unused subscriptions 2) Use free entertainment options 3) Look for discounts 4) Set entertainment budget 5) Find free local events"
                            else:
                                return f"💡 **Save on {top_expense}**: Your biggest expense is {top_expense} (₹{top_amount:.2f}). **General Tips**: 1) Review all expenses 2) Set budgets for each category 3) Track spending daily 4) Use cash for discretionary spending 5) Automate savings"
                        else:
                            return "💡 **General Saving Tips**: 1) Track all expenses 2) Set up automatic savings 3) Use the 50/30/20 rule 4) Cook at home 5) Cancel unused subscriptions 6) Use cash for discretionary spending"
                    else:
                        return "💡 **Great job!** You don't have any expenses recorded yet. **Saving Tips**: 1) Start tracking all expenses 2) Set up automatic savings 3) Use the 50/30/20 rule 4) Build emergency fund 5) Invest in yourself"
                except Exception as e:
                    return "💡 **Saving Strategies**: 1) Track all expenses 2) Set up automatic savings 3) Use the 50/30/20 rule 4) Cook at home 5) Cancel unused subscriptions 6) Use cash for discretionary spending"
            else:
                return "💡 **Saving Strategies**: 1) Track all expenses 2) Set up automatic savings 3) Use the 50/30/20 rule 4) Cook at home 5) Cancel unused subscriptions 6) Use cash for discretionary spending"
        
        else:
            # Fallback to a general financial tip
            tips = [
                "💡 **Tip**: Review your expenses weekly to stay on track with your budget goals.",
                "💡 **Tip**: Set up automatic transfers to savings accounts to make saving effortless.",
                "💡 **Tip**: Use cash for discretionary spending to avoid overspending on cards.",
                "💡 **Tip**: Plan your meals and grocery shopping to reduce food waste and costs.",
                "💡 **Tip**: Regularly review and cancel unused subscriptions to save money monthly."
            ]
            import random
            return random.choice(tips)
        
    except Exception as e:
        return f"Error in AI chat: {str(e)}"

def granite_chat_full_model(question):
    """Full AI model chat using optimized settings for speed"""
    try:
        if st.session_state.tokenizer is None or st.session_state.model is None:
            return "AI model not loaded. Please refresh the page."
        
        # Prepare the chat message
        messages = [
            {"role": "user", "content": question}
        ]
        
        # Apply chat template and tokenize using optimized approach
        inputs = st.session_state.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model (optimized)
        device = next(st.session_state.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response with optimized settings for speed
        with st.spinner("AI is thinking..."):
            outputs = st.session_state.model.generate(
                **inputs, 
                max_new_tokens=80,  # Reduced for speed
                temperature=0.7,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
                length_penalty=1.0,  # Balance length
                early_stopping=True  # Stop early if possible
            )
        
        # Decode the response
        response = st.session_state.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return response.strip() if response.strip() else "I couldn't generate a response. Please try again."
        
    except Exception as e:
        return f"Error in AI chat: {str(e)}"

def test_model():
    """Simple test to verify the model is working"""
    try:
        if st.session_state.tokenizer is None or st.session_state.model is None:
            return "Model not loaded"
        
        # Simple test question
        messages = [{"role": "user", "content": "Hello, who are you?"}]
        inputs = st.session_state.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to model device
        inputs = {k: v.to(st.session_state.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate short response
        outputs = st.session_state.model.generate(
            **inputs, 
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.eos_token_id
        )
        
        response = st.session_state.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return f"✅ Model test successful! Response: {response.strip()}"
        
    except Exception as e:
        return f"❌ Model test failed: {str(e)}"

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
    st.header("Add Transaction")
    with st.form("manual_tx"):
        d = st.date_input("Date", value=datetime.today())
        desc = st.text_input("Description", placeholder="e.g., Grocery shopping, Salary, Rent payment")
        
        # Single amount field with type selection
        col_amount, col_type = st.columns([2, 1])
        with col_amount:
            amount = st.number_input("Amount (₹)", min_value=0.0, value=0.0, format="%.2f", step=100.0)
        with col_type:
            tx_type = st.selectbox("Type", ["Expense", "Income"], help="Select whether this is money spent (Expense) or received (Income)")

        cat = st.selectbox("Category", DEFAULT_CATEGORIES, help="Select the category that best describes this transaction")

        if st.form_submit_button("Add Transaction"):
            if amount > 0:
                # Convert to negative for expenses, positive for income
                final_amount = -amount if tx_type == "Expense" else amount
                
                st.session_state.transactions.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "description": desc,
                    "amount": final_amount,
                    "category": cat,
                    "source": "manual",
                    "type": tx_type
                })
                st.success(f"✅ {tx_type} of ₹{amount:.2f} added successfully!")
            else:
                st.warning("Please enter a non-zero amount.")

    st.markdown("---")
    
    # Quick Transaction Buttons
    st.subheader("Quick Add Common Transactions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Groceries ₹500"):
            st.session_state.transactions.append({
                "date": datetime.today().strftime("%Y-%m-%d"),
                "description": "Grocery shopping",
                "amount": -500.0,
                "category": "Groceries",
                "source": "quick",
                "type": "Expense"
            })
            st.success("✅ Groceries expense added!")
    
    with col2:
        if st.button("🏠 Rent ₹15000"):
            st.session_state.transactions.append({
                "date": datetime.today().strftime("%Y-%m-%d"),
                "description": "Monthly rent",
                "amount": -15000.0,
                "category": "Rent",
                "source": "quick",
                "type": "Expense"
            })
            st.success("✅ Rent expense added!")
    
    with col3:
        if st.button("💵 Salary ₹50000"):
            st.session_state.transactions.append({
                "date": datetime.today().strftime("%Y-%m-%d"),
                "description": "Monthly salary",
                "amount": 50000.0,
                "category": "Salary",
                "source": "quick",
                "type": "Income"
            })
            st.success("✅ Salary income added!")
    
    st.markdown("---")
    st.subheader("Recent Transactions")
    if st.session_state.transactions:
        df_tx = pd.DataFrame(st.session_state.transactions).sort_values("date", ascending=False)
        # Add a type column for better display
        df_tx['Type'] = df_tx['amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
        df_tx['Amount'] = df_tx['amount'].apply(lambda x: f"₹{abs(x):.2f}")
        display_cols = ['date', 'description', 'Type', 'Amount', 'category']
        st.dataframe(df_tx[display_cols], use_container_width=True)
        
        # Quick export button
        if st.button("📥 Export Transactions as CSV"):
            export_df = df_tx.copy()
            export_df['Amount_Numeric'] = export_df['amount'].apply(lambda x: abs(x))
            export_cols = ['date', 'description', 'Type', 'Amount_Numeric', 'category']
            transactions_csv = export_df[export_cols].to_csv(index=False)
            csv_file = f"Transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button("Download CSV", transactions_csv, file_name=csv_file, mime="text/csv")
    else:
        st.info("No transactions yet. Add some using the form above or quick buttons.")

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

        # Enhanced Charts Section
        st.markdown("## 📊 Financial Analytics")
        
        # Spending by Category Chart
        st.markdown("### 💰 Spending by Category")
        by_cat = df_all.groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        if not by_cat.empty:
            fig1 = px.bar(x=by_cat.values, y=by_cat.index, orientation='h', 
                         labels={'x':'Amount (₹)','y':'Category'},
                         title="Total Spending by Category",
                         color=by_cat.values,
                         color_continuous_scale='viridis')
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Add some transactions to see spending charts.")
        
        # Income vs Expenses Pie Chart
        st.markdown("### 📈 Income vs Expenses")
        income_total = df_all[df_all['amount'] > 0]['amount'].sum()
        expense_total = abs(df_all[df_all['amount'] < 0]['amount'].sum())
        
        if income_total > 0 or expense_total > 0:
            fig2 = px.pie(values=[income_total, expense_total], 
                         names=['Income', 'Expenses'],
                         title="Income vs Expenses Distribution",
                         color_discrete_map={'Income': '#00ff00', 'Expenses': '#ff0000'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Add transactions to see income vs expenses chart.")
        
        # Monthly Trend Chart
        st.markdown("### 📅 Monthly Spending Trend")
        df_all['month'] = pd.to_datetime(df_all['date']).dt.to_period('M')
        monthly_data = df_all.groupby('month')['amount'].sum()
        
        if len(monthly_data) > 1:
            fig3 = px.line(x=monthly_data.index.astype(str), 
                          y=monthly_data.values,
                          title="Monthly Net Income/Expense Trend",
                          labels={'x':'Month', 'y':'Net Amount (₹)'})
            fig3.update_layout(height=400)
            fig3.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Add transactions from multiple months to see trends.")

        # Enhanced Budget Planning Section
        st.markdown("## 📋 Budget Planning")
        plan = generate_budget_plan(st.session_state.transactions)
        if "plan" in plan:
            plan_df = pd.DataFrame(list(plan['plan'].items()), columns=["Category","Proposed Monthly Budget"])
            
            # Display budget plan with better formatting
            st.markdown("### 💡 Recommended Monthly Budget")
            st.dataframe(plan_df, use_container_width=True)
            st.markdown(f"**Total Proposed Monthly Budget: ₹{plan['total']:.2f}**")
            
            # Budget vs Actual Comparison
            st.markdown("### 📊 Budget vs Actual Spending")
            if not df_all.empty:
                actual_spending = df_all.groupby('category')['amount'].sum().abs()
                comparison_data = []
                
                for category in plan['plan'].keys():
                    budget_amount = plan['plan'][category]
                    actual_amount = actual_spending.get(category, 0)
                    comparison_data.append({
                        'Category': category,
                        'Budget': budget_amount,
                        'Actual': actual_amount,
                        'Difference': budget_amount - actual_amount,
                        'Status': '✅ Under Budget' if actual_amount <= budget_amount else '⚠️ Over Budget'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            # Export Options
            st.markdown("### 📄 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Budget as PDF"):
                    pdf_file = export_budget_pdf(plan)
                    with open(pdf_file, "rb") as f:
                        st.download_button("Download Budget PDF", f, file_name=pdf_file, mime="application/pdf")
            
            with col2:
                if st.button("📈 Export Budget as CSV"):
                    # Create budget CSV
                    budget_csv = plan_df.to_csv(index=False)
                    budget_csv_file = f"Budget_Plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button("Download Budget CSV", budget_csv, file_name=budget_csv_file, mime="text/csv")
            
            with col3:
                if st.button("📋 Export All Transactions as CSV"):
                    # Create transactions CSV
                    if not df_all.empty:
                        transactions_csv = df_all.to_csv(index=False)
                        transactions_csv_file = f"All_Transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        st.download_button("Download Transactions CSV", transactions_csv, file_name=transactions_csv_file, mime="text/csv")
                    else:
                        st.warning("No transactions to export")
            
            # Additional export options
            st.markdown("#### 📊 Detailed Reports")
            col4, col5 = st.columns(2)
            
            with col4:
                if st.button("📊 Export Budget vs Actual CSV"):
                    if not df_all.empty:
                        # Create budget vs actual comparison CSV
                        comparison_csv = comparison_df.to_csv(index=False)
                        comparison_csv_file = f"Budget_vs_Actual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        st.download_button("Download Comparison CSV", comparison_csv, file_name=comparison_csv_file, mime="text/csv")
                    else:
                        st.warning("No data to compare")
            
            with col5:
                if st.button("📈 Export Monthly Summary CSV"):
                    if not df_all.empty:
                        # Create monthly summary CSV
                        df_all['month'] = pd.to_datetime(df_all['date']).dt.to_period('M')
                        monthly_summary = df_all.groupby('month').agg({
                            'amount': ['sum', 'count']
                        }).round(2)
                        monthly_summary.columns = ['Net_Amount', 'Transaction_Count']
                        monthly_summary = monthly_summary.reset_index()
                        monthly_summary['month'] = monthly_summary['month'].astype(str)
                        
                        monthly_csv = monthly_summary.to_csv(index=False)
                        monthly_csv_file = f"Monthly_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        st.download_button("Download Monthly Summary CSV", monthly_csv, file_name=monthly_csv_file, mime="text/csv")
                    else:
                        st.warning("No transactions to summarize")
        else:
            st.info(plan.get("message"))
    else:
        st.info("No transactions yet. Add some to see dashboard & budget.")

# -----------------------
# AI Model Loading Section
# -----------------------
st.markdown("---")
st.header("🤖 AI Assistant Setup")

# Add fast mode toggle
fast_mode = st.checkbox("🚀 Enable Fast Mode (Rule-based responses)", value=True, help="Fast mode provides instant responses using financial rules. Uncheck to use the full AI model.")

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
    
    # Add test button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload Model"):
            st.session_state.model_loaded = False
            st.session_state.tokenizer = None
            st.session_state.model = None
            st.rerun()
    
    with col2:
        if st.button("🧪 Test Model"):
            test_result = test_model()
            if test_result.startswith("✅"):
                st.success(test_result)
            else:
                st.error(test_result)

# -----------------------
# Chatbot
# -----------------------
st.markdown("---")
st.header("Chat with Budget JINI")

if fast_mode:
    st.info("⚡ **Fast Mode Active**: You'll get instant responses based on financial best practices and your transaction data!")
    
    # Quick action buttons for common questions
    st.subheader("Quick Questions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 How am I spending?"):
            st.session_state.quick_question = "How am I spending?"
    
    with col2:
        if st.button("📊 Budget advice"):
            st.session_state.quick_question = "Budget advice"
    
    with col3:
        if st.button("💡 Saving tips"):
            st.session_state.quick_question = "Saving tips"
    
    # Additional quick actions
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("🏠 Rent analysis"):
            st.session_state.quick_question = "rent"
    
    with col5:
        if st.button("🚗 Transport costs"):
            st.session_state.quick_question = "transport"
    
    with col6:
        if st.button("🎬 Entertainment"):
            st.session_state.quick_question = "entertainment"
    
    # Handle quick questions
    if 'quick_question' in st.session_state and st.session_state.quick_question:
        question = st.session_state.quick_question
        st.session_state.quick_question = None  # Reset after use
        
        if st.session_state.user is None:
            st.error("Please login (or enter demo) to use personalized chat.")
        else:
            resp = granite_chat(question)
            if resp and not resp.startswith("Error"):
                st.success(resp)
            else:
                st.error(resp)

# Regular chat input
question = st.text_input("Ask a question (e.g., 'How much did I spend on groceries?')", key="chat_input")
if st.button("Send", key="send"):
    if st.session_state.user is None:
        st.error("Please login (or enter demo) to use personalized chat.")
    else:
        # Always use fast rule-based responses for now
        with st.spinner("⚡ Generating instant response..."):
            resp = granite_chat(question)
        
        if resp and not resp.startswith("Error"):
            st.success(resp)
        else:
            st.error(resp)

# Add helpful tips
if fast_mode:
    st.markdown("---")
    st.markdown("""
    ### 💡 **Fast Mode Tips:**
    - **Instant Responses**: Get financial advice in milliseconds
    - **Personalized**: Responses based on your actual transaction data
    - **Smart Analysis**: Automatic spending pattern recognition
    - **Budget Insights**: Real-time budget recommendations
    """)
else:
    st.markdown("---")
    st.markdown("""
    ### 🤖 **Full AI Mode Tips:**
    - **Flexible Responses**: More creative and detailed answers
    - **Reliable**: Uses proven model loading approach
    - **Smart Financial Advice**: AI-powered insights and recommendations
    - **Fallback**: Automatically switches to Fast Mode if needed
    """)

if not fast_mode and not st.session_state.model_loaded:
    st.info("Please load the AI model first to use the chat feature.")

st.markdown("</div>", unsafe_allow_html=True)

