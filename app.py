import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import docx
import re

# IMPORTANT:
# Add this to requirements.txt
# sentence-transformers

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Resume ATS by Vikesh",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🚀 AI Resume ATS Scoring Engine")
st.subheader("Version 8B - Semantic Resume Intelligence")
st.caption("Built by Vikesh Sagar Bairam")

st.markdown("""
### Next Generation AI Hiring Platform

This version uses AI embeddings to understand meaning, not just keywords.

### Features

✅ Semantic Resume Matching  
✅ Understands Similar Meaning  
✅ Skills Match  
✅ Education Detection  
✅ Experience Detection  
✅ Role Prediction  
✅ Missing Skills  
✅ Smart Candidate Ranking  
✅ Resume Preview  
✅ CSV Export  

### Creator

👨‍💻 **Vikesh Sagar Bairam**  
📧 vikebairam@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/vikesh-bairam-219769258/

---
""")

# ---------------------------------------------------
# MASTER SKILLS
# ---------------------------------------------------
skills_db = [
    "python","sql","excel","tableau","power bi",
    "machine learning","deep learning","aws","azure",
    "java","cloud","pandas","numpy","statistics",
    "etl","spark","tensorflow","leadership",
    "communication","project management",
    "docker","kubernetes","git","linux",
    "react","node.js","api","microservices"
]

# ---------------------------------------------------
# FILE READER
# ---------------------------------------------------
def extract_text(uploaded_file):

    filename = uploaded_file.name.lower()
    text = ""

    try:
        if filename.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "

        elif filename.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + " "

        elif filename.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")

    except:
        text = ""

    return text.lower()

# ---------------------------------------------------
# FIND SKILLS
# ---------------------------------------------------
def find_skills(text):
    found = []

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found

# ---------------------------------------------------
# EDUCATION DETECTION
# ---------------------------------------------------
def detect_education(text):

    education = []

    if "bachelor" in text or "b.tech" in text or "bs " in text:
        education.append("Bachelor Degree")

    if "master" in text or "mba" in text or "ms " in text:
        education.append("Masters Degree")

    if "phd" in text:
        education.append("PhD")

    return ", ".join(education)

# ---------------------------------------------------
# EXPERIENCE DETECTION
# ---------------------------------------------------
def detect_experience(text):

    pattern = r'(\d+)\+?\s+years'

    match = re.findall(pattern, text)

    if match:
        years = max([int(x) for x in match])
        return f"{years}+ Years Experience"

    return "Experience Not Clear"

# ---------------------------------------------------
# ROLE PREDICTION
# ---------------------------------------------------
def predict_role(text):

    if "machine learning" in text or "tensorflow" in text:
        return "Data Scientist"

    elif "sql" in text and "tableau" in text:
        return "Data Analyst"

    elif "aws" in text or "docker" in text:
        return "Cloud / DevOps Engineer"

    elif "java" in text or "api" in text or "microservices" in text:
        return "Software Engineer"

    elif "project management" in text:
        return "Project Manager"

    return "General Professional"

# ---------------------------------------------------
# SEMANTIC SCORE
# ---------------------------------------------------
def semantic_similarity(resume_text, jd_text):

    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(jd_text, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()

    return score * 100

# ---------------------------------------------------
# SMART ATS SCORE
# ---------------------------------------------------
def final_score(resume_text, jd_text):

    jd_skills = set(find_skills(jd_text))
    resume_skills = set(find_skills(resume_text))

    matched = jd_skills.intersection(resume_skills)
    missing = jd_skills - resume_skills

    # Skill Score
    if len(jd_skills) > 0:
        skill_score = (len(matched) / len(jd_skills)) * 100
    else:
        skill_score = 0

    # Semantic Score
    semantic_score = semantic_similarity(
        resume_text[:4000],
        jd_text[:4000]
    )

    # Education
    edu = detect_education(resume_text)
    edu_score = 10 if edu else 0

    # Final Score
    total = (
        skill_score * 0.40 +
        semantic_score * 0.40 +
        edu_score * 0.20
    )

    if total > 100:
        total = 100

    return round(total,2), matched, missing

# ---------------------------------------------------
# FIT STATUS
# ---------------------------------------------------
def fit_status(score):

    if score >= 80:
        return "🔥 STRONG FIT"

    elif score >= 65:
        return "✅ GOOD FIT"

    elif score >= 50:
        return "👍 POSSIBLE FIT"

    return "❌ NOT FIT"

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
uploaded_files = st.file_uploader(
    "📄 Upload Resume Files",
    type=["pdf","docx","txt"],
    accept_multiple_files=True
)

job_desc = st.text_area(
    "📝 Paste Job Description",
    height=220
)

analyze = st.button(
    "🚀 Analyze Candidates",
    use_container_width=True
)

# ---------------------------------------------------
# PROCESS
# ---------------------------------------------------
if analyze:

    if not uploaded_files:
        st.warning("Please upload resumes.")

    elif job_desc.strip() == "":
        st.warning("Please paste job description.")

    else:

        results = []

        with st.spinner("Running AI semantic analysis..."):

            for file in uploaded_files:

                resume_text = extract_text(file)

                if resume_text.strip() == "":
                    continue

                score, matched, missing = final_score(
                    resume_text,
                    job_desc.lower()
                )

                results.append({
                    "Candidate": file.name,
                    "Score": score,
                    "Role Prediction": predict_role(resume_text),
                    "Education": detect_education(resume_text),
                    "Experience": detect_experience(resume_text),
                    "Matched Skills": ", ".join(matched),
                    "Missing Skills": ", ".join(missing),
                    "Fit Status": fit_status(score),
                    "Resume Text": resume_text
                })

        results = sorted(
            results,
            key=lambda x: x["Score"],
            reverse=True
        )

        df = pd.DataFrame(results)

        # KPIs
        c1,c2,c3 = st.columns(3)

        with c1:
            st.metric("🏆 Top Candidate", results[0]["Candidate"])

        with c2:
            st.metric("📈 Top Score", f"{results[0]['Score']}%")

        with c3:
            st.metric("📊 Avg Score", f"{round(df['Score'].mean(),2)}%")

        st.markdown("---")

        # Candidate Cards
        for i,row in enumerate(results,1):

            st.markdown(f"## #{i} {row['Candidate']}")

            st.progress(int(row["Score"]))

            st.success(
                f"{row['Score']}% | {row['Fit Status']}"
            )

            st.write(f"**Predicted Role:** {row['Role Prediction']}")
            st.write(f"**Education:** {row['Education']}")
            st.write(f"**Experience:** {row['Experience']}")
            st.write(f"**Matched Skills:** {row['Matched Skills']}")
            st.write(f"**Missing Skills:** {row['Missing Skills']}")

            with st.expander("📄 View Resume"):
                st.text_area(
                    "Resume Preview",
                    row["Resume Text"][:7000],
                    height=350
                )

            st.markdown("---")

        # Dashboard Table
        st.subheader("📋 Recruiter Dashboard")

        show_df = df.drop(columns=["Resume Text"])

        st.dataframe(show_df, use_container_width=True)

        # Chart
        st.subheader("📊 Candidate Score Chart")

        fig, ax = plt.subplots()

        ax.bar(show_df["Candidate"], show_df["Score"])
        ax.set_ylabel("Score %")
        ax.set_xlabel("Candidate")

        plt.xticks(rotation=45, ha="right")

        st.pyplot(fig)

        # Download
        csv = show_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Candidate Report",
            csv,
            "candidate_report.csv",
            "text/csv",
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