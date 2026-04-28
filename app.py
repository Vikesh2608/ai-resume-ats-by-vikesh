import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Resume ATS by Vikesh",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🚀 AI Resume ATS Scoring Engine")
st.subheader("Version 2 - Multi Candidate Ranking")
st.caption("Built by Vikesh")

st.markdown("""
### Why This Tool Exists

Recruiters receive hundreds of resumes for one role.  
This AI tool helps quickly shortlist the best candidates.

### Features

✅ Upload Multiple Resumes  
✅ Candidate Ranking  
✅ ATS Match Score  
✅ Faster Hiring Decisions  
✅ Recruiter Friendly Dashboard

---
""")

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

# Extract text from PDF
def extract_text(uploaded_file):
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text.lower()


# Calculate score
def calculate_score(resume_text, jd_text):

    docs = [resume_text, jd_text]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(docs)

    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return round(score * 100, 2)


# Skill database
skills_db = [
    "python", "sql", "excel", "tableau", "power bi",
    "machine learning", "deep learning", "aws", "azure",
    "java", "cloud", "pandas", "numpy", "statistics",
    "etl", "spark", "tensorflow", "leadership",
    "communication", "project management"
]


# Detect skills
def find_skills(text):

    found = []

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found


# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "📄 Upload Multiple Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_desc = st.text_area(
    "📝 Paste Job Description",
    height=220,
    placeholder="Paste full job description here..."
)

analyze = st.button(
    "🚀 Rank Candidates",
    use_container_width=True
)

# ---------------------------------------------------
# PROCESSING
# ---------------------------------------------------

if analyze:

    if not uploaded_files:
        st.warning("Please upload at least one resume.")

    elif job_desc.strip() == "":
        st.warning("Please paste the job description.")

    else:

        jd_text = job_desc.lower()
        results = []

        with st.spinner("Ranking candidates..."):

            for file in uploaded_files:

                resume_text = extract_text(file)

                score = calculate_score(
                    resume_text,
                    jd_text
                )

                skills = find_skills(resume_text)

                results.append({
                    "Candidate": file.name,
                    "Score": score,
                    "Skills": ", ".join(skills)
                })

        # Sort descending
        results = sorted(
            results,
            key=lambda x: x["Score"],
            reverse=True
        )

        # ---------------------------------------------------
        # TOP RESULTS
        # ---------------------------------------------------
        st.subheader("🏆 Candidate Rankings")

        for idx, row in enumerate(results, start=1):

            st.markdown(
                f"### #{idx} {row['Candidate']}"
            )

            st.progress(int(row["Score"]))
            st.success(
                f"ATS Match Score: {row['Score']}%"
            )

            st.write(
                f"**Detected Skills:** {row['Skills']}"
            )

            st.markdown("---")

        # ---------------------------------------------------
        # DATA TABLE
        # ---------------------------------------------------
        st.subheader("📊 Ranking Table")

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True
        )

        # ---------------------------------------------------
        # CSV DOWNLOAD
        # ---------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Rankings CSV",
            data=csv,
            file_name="candidate_rankings.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.caption("© 2026 Built by Vikesh | AI Resume ATS Engine")