import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Resume ATS by Vikesh",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------------
# HEADER
# -----------------------------------
st.title("🚀 AI Resume ATS Scoring Engine")
st.subheader("Built by Vikesh")

st.markdown("""
### Why This Tool Exists

Hiring teams receive hundreds of resumes for one role.  
This AI-powered tool helps recruiters and job seekers quickly evaluate:

✅ Resume Match Score  
✅ Missing Skills Detection  
✅ Matching Skills Found  
✅ Better Hiring Decisions  
✅ Faster Resume Screening  

---
""")

# -----------------------------------
# FUNCTIONS
# -----------------------------------

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


# Skills database
skills_db = [
    "python", "sql", "excel", "tableau", "power bi",
    "machine learning", "deep learning", "aws", "azure",
    "java", "cloud", "pandas", "numpy", "statistics",
    "data analysis", "etl", "spark", "tensorflow",
    "communication", "leadership", "project management"
]


# Find matching skills
def find_skills(text):
    found = []

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found


# -----------------------------------
# INPUT SECTION
# -----------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload Resume PDF",
    type=["pdf"]
)

job_desc = st.text_area(
    "📝 Paste Job Description",
    height=220,
    placeholder="Paste the full job description here..."
)

analyze = st.button(
    "🚀 Analyze Resume",
    use_container_width=True
)

# -----------------------------------
# PROCESSING
# -----------------------------------

if analyze:

    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")
    
    elif job_desc.strip() == "":
        st.warning("Please paste the job description.")

    else:

        with st.spinner("Analyzing Resume..."):

            resume_text = extract_text(uploaded_file)
            jd_text = job_desc.lower()

            score = calculate_score(resume_text, jd_text)

            resume_skills = set(find_skills(resume_text))
            jd_skills = set(find_skills(jd_text))

            matched_skills = list(resume_skills.intersection(jd_skills))
            missing_skills = list(jd_skills - resume_skills)

        # -----------------------------
        # SCORE SECTION
        # -----------------------------
        st.subheader("📊 ATS Match Score")

        st.progress(int(score))
        st.success(f"{score}% Match")

        # -----------------------------
        # TWO COLUMNS
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matching Skills")

            if matched_skills:
                for skill in matched_skills:
                    st.write("✔️", skill.title())
            else:
                st.write("No matching skills found.")

        with col2:
            st.subheader("⚠️ Missing Skills")

            if missing_skills:
                for skill in missing_skills:
                    st.write("❌", skill.title())
            else:
                st.write("No major skill gaps.")

        # -----------------------------
        # RESUME PREVIEW
        # -----------------------------
        st.subheader("📄 Resume Preview")
        st.write(resume_text[:3000])

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")
st.caption("© 2026 Built by Vikesh | AI Resume ATS Scoring Engine")