import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
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
st.subheader("Version 3 - Executive Recruiter Dashboard")
st.caption("Built by Vikesh Sagar Bairam")

st.markdown("""
### Intelligent Candidate Screening Platform

This AI-powered platform helps recruiters shortlist candidates faster using NLP.

### Premium Features

✅ Upload Multiple Resumes  
✅ Candidate Ranking  
✅ Top Candidate Highlight  
✅ Interview Questions Generator  
✅ Hiring Recommendation  
✅ Recruiter Analytics Dashboard  
✅ CSV Export  

### Creator

👨‍💻 **Vikesh Sagar Bairam**  
📧 vikebairam@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/vikesh-bairam-219769258/

---
""")

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

# Extract text from uploaded PDF
def extract_text(uploaded_file):
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text.lower()


# Calculate similarity score
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


# Hiring recommendation
def hiring_recommendation(score):

    if score >= 80:
        return "🔥 Strong Hire"

    elif score >= 60:
        return "👍 Consider"

    else:
        return "⚠️ Weak Fit"


# Generate interview questions
def interview_questions(skills):

    questions = []

    if "python" in skills:
        questions.append("Explain a Python project you built.")

    if "sql" in skills:
        questions.append("How do you optimize SQL queries?")

    if "aws" in skills:
        questions.append("Describe your AWS cloud experience.")

    if "leadership" in skills:
        questions.append("Tell us about a leadership challenge.")

    if "machine learning" in skills:
        questions.append("Describe a machine learning model you built.")

    if not questions:
        questions.append("Tell us about your background and strengths.")

    return questions


# ---------------------------------------------------
# INPUT SECTION
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
    "🚀 Analyze Candidates",
    use_container_width=True
)

# ---------------------------------------------------
# PROCESSING
# ---------------------------------------------------

if analyze:

    if not uploaded_files:
        st.warning("Please upload resumes.")

    elif job_desc.strip() == "":
        st.warning("Please paste the job description.")

    else:

        jd_text = job_desc.lower()

        results = []

        with st.spinner("Analyzing candidates..."):

            for file in uploaded_files:

                resume_text = extract_text(file)

                score = calculate_score(
                    resume_text,
                    jd_text
                )

                skills = find_skills(resume_text)

                recommendation = hiring_recommendation(score)

                results.append({
                    "Candidate": file.name,
                    "Score": score,
                    "Skills": ", ".join(skills),
                    "Recommendation": recommendation
                })

        # Sort Results
        results = sorted(
            results,
            key=lambda x: x["Score"],
            reverse=True
        )

        df = pd.DataFrame(results)

        # ---------------------------------------------------
        # KPI SECTION
        # ---------------------------------------------------
        top_candidate = results[0]["Candidate"]
        top_score = results[0]["Score"]
        avg_score = round(df["Score"].mean(), 2)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🏆 Top Candidate", top_candidate)

        with c2:
            st.metric("📈 Top Score", f"{top_score}%")

        with c3:
            st.metric("📊 Avg Score", f"{avg_score}%")

        st.markdown("---")

        # ---------------------------------------------------
        # RANKINGS
        # ---------------------------------------------------
        st.subheader("🏆 Candidate Rankings")

        for idx, row in enumerate(results, start=1):

            st.markdown(f"### #{idx} {row['Candidate']}")

            st.progress(int(row["Score"]))

            st.success(
                f"{row['Score']}% Match | {row['Recommendation']}"
            )

            st.write(
                f"**Detected Skills:** {row['Skills']}"
            )

            qs = interview_questions(row["Skills"].lower())

            st.write("**Suggested Interview Questions:**")

            for q in qs:
                st.write("•", q)

            st.markdown("---")

        # ---------------------------------------------------
        # TABLE
        # ---------------------------------------------------
        st.subheader("📋 Recruiter Dashboard")

        st.dataframe(
            df,
            use_container_width=True
        )

        # ---------------------------------------------------
        # CHART
        # ---------------------------------------------------
        st.subheader("📊 Candidate Score Chart")

        fig, ax = plt.subplots()

        ax.bar(df["Candidate"], df["Score"])
        ax.set_ylabel("Score %")
        ax.set_xlabel("Candidate")
        plt.xticks(rotation=45, ha="right")

        st.pyplot(fig)

        # ---------------------------------------------------
        # DOWNLOAD REPORT
        # ---------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Candidate Report",
            data=csv,
            file_name="candidate_report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption(
    "© 2026 Built by Vikesh Sagar Bairam | "
    "LinkedIn: linkedin.com/in/vikesh-bairam-219769258 | "
    "Email: vikebairam@gmail.com"
)