import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import docx
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
st.subheader("Version 5 - Universal Resume Upload Platform")
st.caption("Built by Vikesh Sagar Bairam")

st.markdown("""
### Intelligent Candidate Screening Platform

Upload resumes in multiple formats and shortlist candidates instantly.

### Premium Features

✅ Upload PDF / DOCX / TXT Resumes  
✅ Multiple Candidate Ranking  
✅ Missing Skills Detection  
✅ Good Fit Badge  
✅ Interview Questions Generator  
✅ Recruiter Dashboard  
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

# Extract text from multiple file formats
def extract_text(uploaded_file):

    filename = uploaded_file.name.lower()
    text = ""

    try:

        # PDF
        if filename.endswith(".pdf"):

            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + " "

        # DOCX
        elif filename.endswith(".docx"):

            doc = docx.Document(uploaded_file)

            for para in doc.paragraphs:
                text += para.text + " "

        # TXT
        elif filename.endswith(".txt"):

            text = uploaded_file.read().decode("utf-8")

        else:
            text = ""

    except:
        text = ""

    return text.lower()


# Calculate ATS score
def calculate_score(resume_text, jd_text):

    docs = [resume_text, jd_text]

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(docs)

    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return round(score * 100, 2)


# Skills database
skills_db = [
    "python", "sql", "excel", "tableau", "power bi",
    "machine learning", "deep learning", "aws", "azure",
    "java", "cloud", "pandas", "numpy", "statistics",
    "etl", "spark", "tensorflow", "leadership",
    "communication", "project management"
]


# Find skills
def find_skills(text):

    found = []

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found


# Fit Status
def fit_status(score, missing_count):

    if score >= 75 and missing_count <= 2:
        return "✅ GOOD FIT"

    elif score >= 60:
        return "👍 POSSIBLE FIT"

    else:
        return "❌ NOT FIT"


# Interview Questions
def interview_questions(skills):

    questions = []

    if "python" in skills:
        questions.append("Explain a Python project you built.")

    if "sql" in skills:
        questions.append("How do you optimize SQL queries?")

    if "aws" in skills:
        questions.append("Describe your AWS cloud experience.")

    if "machine learning" in skills:
        questions.append("Describe a machine learning model.")

    if "leadership" in skills:
        questions.append("Tell us about your leadership experience.")

    if not questions:
        questions.append("Tell us about your background.")

    return questions


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "📄 Upload Resume Files",
    type=["pdf", "docx", "txt"],
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
        jd_skills = set(find_skills(jd_text))

        results = []

        with st.spinner("Analyzing candidates..."):

            for file in uploaded_files:

                resume_text = extract_text(file)

                if resume_text.strip() == "":
                    continue

                score = calculate_score(
                    resume_text,
                    jd_text
                )

                resume_skills = set(find_skills(resume_text))

                matched_skills = list(
                    resume_skills.intersection(jd_skills)
                )

                missing_skills = list(
                    jd_skills - resume_skills
                )

                status = fit_status(
                    score,
                    len(missing_skills)
                )

                results.append({
                    "Candidate": file.name,
                    "Score": score,
                    "Matched Skills": ", ".join(matched_skills),
                    "Missing Skills": ", ".join(missing_skills),
                    "Fit Status": status
                })

        # Sort Results
        results = sorted(
            results,
            key=lambda x: x["Score"],
            reverse=True
        )

        if len(results) == 0:
            st.error("No readable resumes found.")
            st.stop()

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
        # CANDIDATE RESULTS
        # ---------------------------------------------------
        st.subheader("🏆 Candidate Rankings")

        for idx, row in enumerate(results, start=1):

            st.markdown(f"### #{idx} {row['Candidate']}")

            st.progress(int(row["Score"]))

            st.success(
                f"{row['Score']}% Match | {row['Fit Status']}"
            )

            st.write(
                f"**Matched Skills:** {row['Matched Skills']}"
            )

            st.write(
                f"**Missing Skills:** {row['Missing Skills']}"
            )

            qs = interview_questions(
                row["Matched Skills"].lower()
            )

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