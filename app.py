import streamlit as st
import pickle
import re
from collections import Counter
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocess
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Spam Detection", "History & Analysis", "Model Info"])

# Dark Mode
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown(
        "<style>body {background-color: black; color: white;}</style>",
        unsafe_allow_html=True
    )

# Store history
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- PAGE 1 ----------------
if page == "Spam Detection":

    st.title("📧 Spam Detection System")

    st.write("💡 Try examples:")
    st.code("Win a free iPhone now!!!")
    st.code("Hey, are you coming to class?")

    user_input = st.text_area("Enter your message:")

    if user_input:

        st.info("🔍 Analyzing message...")

        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])

        prediction = model.predict(vectorized)

        # Message stats
        st.write("📏 Message Length:", len(user_input))
        st.write("📝 Word Count:", len(user_input.split()))

        # Probability
        proba = model.predict_proba(vectorized)
        spam_prob = proba[0][1]
        ham_prob = proba[0][0]

        # Result
        if prediction[0] == 1:
            st.error("🚫 Spam Message ❌")
        else:
            st.success("✅ Not Spam ✔️")

        # Progress bar
        st.write("📊 Spam Confidence")
        st.progress(int(spam_prob * 100))

        st.write(f"Spam Probability: {spam_prob:.2f}")
        st.write(f"Ham Probability: {ham_prob:.2f}")

        # Processed text
        st.write("🔧 Processed Text:", processed)

        # Keyword detection
        spam_words = ["win", "free", "prize", "money", "offer"]
        found_words = [word for word in spam_words if word in user_input.lower()]

        if found_words:
            st.warning(f"⚠️ Suspicious words: {found_words}")

        # Top words
        words = processed.split()
        common_words = Counter(words).most_common(3)
        st.write("🔑 Top Words:", common_words)

        # Save history
        st.session_state.history.append((user_input, prediction[0]))

# ---------------- PAGE 2 ----------------
elif page == "History & Analysis":

    st.title("📊 History & Analysis")

    if len(st.session_state.history) == 0:
        st.write("No history yet.")
    else:
        # ✅ Initialize counts
        spam_count = 0
        ham_count = 0
        report = ""

        # ✅ Loop through history
        for msg, pred in st.session_state.history:
            if pred == 1:
                spam_count += 1
                st.write(msg, "→ 🚫 Spam")
                report += f"{msg} → Spam\n"
            else:
                ham_count += 1
                st.write(msg, "→ ✅ Not Spam")
                report += f"{msg} → Not Spam\n"

        # Summary
        st.subheader("📈 Summary")
        st.write("Spam:", spam_count)
        st.write("Not Spam:", ham_count)

        # ✅ BAR CHART (FIXED)
        labels = ['Spam', 'Not Spam']
        values = [spam_count, ham_count]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, values)

        ax.set_title("Spam vs Not Spam Messages")
        ax.set_xlabel("Message Type")
        ax.set_ylabel("Count")

        # Values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    str(height), ha='center', va='bottom')

        ax.grid(True)
        st.pyplot(fig)

        # PIE CHART
        fig2, ax2 = plt.subplots()
        ax2.pie(values, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig2)

        # Download report
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="spam_report.txt",
            mime="text/plain"
        )

        # Clear history
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.success("History Cleared!")

# ---------------- PAGE 3 ----------------
elif page == "Model Info":

    st.title("ℹ️ Model Information")

    st.write("Model: Multinomial Naive Bayes")
    st.write("Vectorizer: TF-IDF")
    st.write("Features: N-grams (1,2)")
    st.write("Dataset: SMS Spam Collection")

    st.write("📌 Features:")
    st.write("- Real-time prediction")
    st.write("- Probability scoring")
    st.write("- Message analysis")
    st.write("- Keyword detection")
    st.write("- Bar & Pie charts")
    st.write("- Multi-page UI")