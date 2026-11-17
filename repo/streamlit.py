
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# CASEPATH UGANDA
# ----------------------------------

st.set_page_config(
    page_title="CasePath Uganda",
    page_icon="⚖️",
    layout="wide"
)

# Load the complete trained system
@st.cache_resource
def load_casepath():
   from pathlib import Path

bundle_path = Path(__file__).parent / "casepath_uganda_complete_system.pkl"
bundle = joblib.load(bundle_path)
    return bundle

try:
    bundle = load_casepath()

    classifier = bundle["classifier"]
    classification_vectorizer = bundle["classification_vectorizer"]
    retrieval_vectorizer = bundle["retrieval_vectorizer"]
    case_vectors = bundle["case_vectors"]
    retrieval_data = bundle["retrieval_data"]
    duration_statistics = bundle["duration_statistics"]

except Exception as e:
    st.error(f"Could not load the CasePath system: {e}")
    st.stop()


# ----------------------------------
# TEXT PREPROCESSING
# ----------------------------------

def preprocess_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text


# ----------------------------------
# CASE RETRIEVAL
# ----------------------------------

def retrieve_similar_cases(description, predicted_category, top_n=5):

    cleaned_text = preprocess_text(description)

    query_vector = retrieval_vectorizer.transform([cleaned_text])

    similarity_scores = cosine_similarity(
        query_vector,
        case_vectors
    ).flatten()

    results = retrieval_data.copy()

    results["similarity"] = similarity_scores * 100

    matching_category = results[
        results["legal_issue_category"] == predicted_category
    ].copy()

    matching_category = matching_category.sort_values(
        by="similarity",
        ascending=False
    )

    if len(matching_category) < top_n:

        other_cases = results[
            results["legal_issue_category"] != predicted_category
        ].copy()

        other_cases = other_cases.sort_values(
            by="similarity",
            ascending=False
        )

        final_results = pd.concat(
            [matching_category, other_cases]
        ).head(top_n)

    else:
        final_results = matching_category.head(top_n)

    return final_results


# ----------------------------------
# HISTORICAL DURATION STATISTICS
# ----------------------------------

def get_duration_statistics(category):

    stats = duration_statistics[
        duration_statistics.index == category
    ]

    if stats.empty:
        return None

    return stats.iloc[0]


# ----------------------------------
# CASEPATH INTERFACE
# ----------------------------------

st.title("⚖️ CasePath Uganda")

st.subheader("Machine Learning-Based Land Case Support System")

st.write(
    """
    CasePath Uganda helps users explore land-related legal concerns
    through machine learning classification, retrieval of similar
    historical cases and historical case-duration statistics.

    This is a research prototype, not a substitute for professional
    legal advice.
    """
)

st.divider()

st.header("Describe Your Land Concern")

user_description = st.text_area(
    "Enter your land-related concern:",
    placeholder="Example: Another person is claiming ownership of land that I purchased...",
    height=150
)

if st.button("Analyse My Concern", type="primary"):

    if not user_description.strip():

        st.warning("Please enter a description first.")

    else:

        cleaned_text = preprocess_text(user_description)

        # Classification
        text_vector = classification_vectorizer.transform(
            [cleaned_text]
        )

        predicted_category = classifier.predict(
            text_vector
        )[0]

        st.success("Analysis completed!")

        st.subheader("1. Predicted Legal Issue Category")

        st.write(f"**{predicted_category.replace('_', ' ').title()}**")

        # Similar cases
        st.subheader("2. Similar Historical Cases")

        similar_cases = retrieve_similar_cases(
            user_description,
            predicted_category
        )

        if similar_cases.empty:
            st.info("No similar cases were found.")

        else:
            for _, case in similar_cases.iterrows():

                st.markdown(
                    f"**{case['case_title_parties']}**"
                )

                st.write(
                    f"Citation: {case['neutral_citation']}"
                )

                st.write(
                    f"Category: {case['legal_issue_category']}"
                )

                st.write(
                    f"Text similarity: {case['similarity']:.2f}%"
                )

                link = case.get("ulii_link", "")

                if pd.notna(link) and str(link).startswith("http"):
                    st.markdown(
                        f"[View original judgment]({link})"
                    )

                st.divider()

        # Historical duration
        st.subheader("3. Historical Case Duration")

        stats = get_duration_statistics(predicted_category)

        if stats is not None:

            st.write(
                f"Historical cases analysed: "
                f"**{int(stats['Number_of_Cases'])}**"
            )

            st.write(
                f"Median duration: "
                f"**{stats['Median_Duration']:.2f} years**"
            )

            st.write(
                f"Average duration: "
                f"**{stats['Average_Duration']:.2f} years**"
            )

            st.caption(
                "These figures describe historical completed cases "
                "in this category. They are not predictions or "
                "guarantees about the duration of an individual case."
            )

        else:
            st.info(
                "Historical duration statistics are unavailable "
                "for this category."
            )

st.divider()

st.caption(
    "CasePath Uganda | Academic Machine Learning Project"
)
