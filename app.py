import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load saved ML components
# -----------------------------

tfidf = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
data = joblib.load("scholarship_data.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Scholarship Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎓 Scholarship Recommendation System")

st.write(
    "Enter your requirements below and the system will recommend "
    "the most relevant scholarships."
)


# -----------------------------
# Student Inputs
# -----------------------------

st.subheader("👨‍🎓 Student Information")

degree = st.selectbox(
    "Degree / Study Level",
    [
        "College freshman",
        "College sophomore",
        "College junior",
        "College senior",
        "Graduate student",
        "Doctoral student"
    ]
)

field = st.text_input(
    "📚 Field of Study",
    placeholder="Example: Computer Science"
)

location = st.text_input(
    "🌎 Preferred Location",
    placeholder="Example: United States"
)

requirements = st.text_area(
    "📝 Additional Requirements",
    placeholder="Example: I need financial assistance for my studies."
)


# -----------------------------
# Recommendation
# -----------------------------

if st.button("🔍 Recommend Scholarships"):

    if field.strip() == "":
        st.warning("Please enter your field of study.")
    else:

        # Combine student requirements
        user_input = (
            degree + " " +
            field + " " +
            location + " " +
            requirements
        )

        # Convert input into TF-IDF
        user_vector = tfidf.transform([user_input])

        # Calculate similarity
        similarity_scores = cosine_similarity(
            user_vector,
            tfidf_matrix
        ).flatten()

        # Get top 5
        top_indices = similarity_scores.argsort()[-5:][::-1]

        st.subheader("🏆 Recommended Scholarships")

        for rank, index in enumerate(top_indices, start=1):

            scholarship = data.iloc[index]

            score = round(
                similarity_scores[index] * 100,
                2
            )

            st.markdown(f"### {rank}. {scholarship['Scholarship Name']}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"💰 **Amount:** {scholarship['Amount']}"
                )

                st.write(
                    f"🌎 **Location:** {scholarship['Location']}"
                )

                st.write(
                    f"🎓 **Eligible Years:** {scholarship['Years']}"
                )

            with col2:
                st.write(
                    f"📊 **Match Score:** {score}%"
                )

                st.write(
                    f"🔗 [Apply / View Scholarship]({scholarship['Link']})"
                )

            st.divider()