import streamlit as st
import json
import os
import openai
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuration
OPENAI_API_KEY = st.secrets["credentials"]["OPENAI_API_KEY"]
JSON_FILE = "proposals.json"
openai.api_key = OPENAI_API_KEY

# Load & Save Proposals
def load_proposals():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

def save_proposals(proposals):
    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(proposals, file, indent=4)

proposals = load_proposals()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

def train_vectorizer(filtered):
    if not filtered:
        return None, None
    docs = [p["problem"] for p in filtered]
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix

# Similar Proposal Finder

def find_similar_proposals(jd_text, top_n=3):
    if not proposals:
        return []

    _, tfidf_matrix = train_vectorizer(proposals)
    jd_vec = vectorizer.transform([jd_text])
    scores = cosine_similarity(jd_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(proposals[i], scores[i]) for i in top_indices if scores[i] > 0.1]

def extract_skills_with_llm(jd_text):
    prompt = f"""
    From the following job description, extract only relevant skills and organize them into clean sections.

    You must:
    - Only show a section (Hard Skills, Tools & Platforms, or Soft Skills) if it is clearly relevant in the job description.
    - Use this exact section format if applicable:
        🔹 Hard Skills
        🔹 Tools & Platforms
        🔹 Soft Skills
    - Use bullet points under each section.
    - Don't include any empty or placeholder sections.
    - No headings or explanation outside the bullet sections.

    Job Description:
    {jd_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600
    )

    skills_text = response.choices[0].message.content.strip()
    return skills_text

def generate_hook_paragraph(jd_text):
    prompt = f"""
    Given the job description below, write a strong hook paragraph (4-5 lines max) that combines the job's main goal and the top skill areas required.
    The hook should:
    - Sound client-focused
    - Show strategic and consultative thinking
    - Not be generic or salesy
    - Mix the core job need and required expertise naturally

    Job Description:
    {jd_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()



# Proposal Generator
def generate_proposal(jd_text, similar,tone="Humanized"):
    portfolio_link = similar[0][0]['portfolio_link'] if similar else ""
    case_study = "".join([
        f"""
        - Project: {proj['client_name']} ({proj['industry']})
        Problem: {proj.get('problem', '')}
        Our Solution: {proj.get('solution', '')}
        Results Achieved: {proj.get('outcome', '')}
        Project Link: {proj.get('project_link', '')}
        """
        for proj, _ in similar[:1]
    ]) if similar else ""

    if tone == "Humanized":
        language_guidelines = """
        - Use plain, friendly English (8th to 10th grade reading level).
        - Soften formal words: say “I’ve worked on…” instead of “I specialize in…”, and “I make sure…” instead of “I ensure…”.
        - Avoid resume-style language. Make it sound like a helpful expert having a conversation.
        - Be warm, honest, and confident — not robotic or salesy.
        """
    else:
        language_guidelines = """
        - Use confident, professional business language.
        - Be clear and concise but keep a natural tone — no buzzwords or filler.
        - Keep the tone consultative and outcome-focused.
        """
  

    first_line = jd_text.strip().split("\n")[0] if jd_text.strip() else "your project requirements"
    # Detect tool preference from JD
    jd_lower = jd_text.lower()
    if "power bi" in jd_lower and "tableau" in jd_lower:
        expert_intro = "My name is Tarsem Singh, and I am a Certified Power BI expert with 8+ years of hands-on development experience..."
    elif "tableau" in jd_lower:
        expert_intro = "My name is Tarsem Singh, and I am a Certified Tableau expert with 8+ years of hands-on development experience..."
    else:
        expert_intro = "My name is Tarsem Singh, and I am a Certified Power BI expert with 8+ years of hands-on development experience..."
    prompt = f"""
    You are an expert proposal writer generating a human-like, natural, and customized proposal based on the job description below.

    - Write a flowing narrative that sounds human and consultative — not robotic.
    - Start by introducing the applicant: "{expert_intro}"
    - Immediately follow that with a 4–5 line custom hook that reflects the client's real needs and highlights relevant hard/soft skills based on the JD.
    - Blend the hook naturally with the rest of the first paragraph — do not write it as a separate section or bullet points.
    - Highlight a matching case study if available (use the format given below), and if a project link is present, include it naturally in the proposal using plain language like: "You can explore this case study here: [link]".
    - Emphasize practical technical skills: Power BI, data modeling (star/snowflake), DAX, Power Query, Azure, SQL, ETL pipelines, and requirement gathering.
    - Focus on how the developer will ensure clean architecture, collaboration with consultants, and meaningful reporting.
     {language_guidelines}
    - End with the portfolio link and signature: "Best regards, Tarsem Singh"

    The client's job description starts with: "{first_line}"

    Here’s a relevant project to reference:
    {case_study}

    The full job description is:
    {jd_text}

    **Must-Haves:**
    - Start with "Dear Client," and introduce yourself immediately.
    - Include a custom intro hook with key skills from the JD.
    - Mention no more than 1–2 projects using Problem-Solution-Outcome.
    - Use ONLY verified past projects from the data provided.
    - End with: "You can view my portfolio here: {portfolio_link} \n\nBest regards, Tarsem Singh"
    - Limit to 300 words.

    **Prohibited:**
    - No fake stats or stories.
    - No generic phrases or templated filler.
    - No headings or bullet points in the final proposal.
    - No assumptions beyond the job description.
    """

    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(prompt))

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip(), token_count


# Streamlit UI
st.set_page_config(layout="wide")
col1, col2 = st.columns([8, 4])

with col1:
    st.title("🤖 AI Proposal Architect")
    tab1, tab2 = st.tabs(["📑 Generate Proposal", "📂 Add Project Data"])

    with tab1:

        jd_text = st.text_area("Paste client's project requirements:", height=250)
        tone = st.radio(
            label="",
            options=["Humanized", "Formal"],
            horizontal=True,
            label_visibility="collapsed"
        )
        if st.button("Generate Proposal"):
            if jd_text.strip():
                similar = find_similar_proposals(jd_text)
                if similar:
                    proposal, tokens = generate_proposal(jd_text, similar, tone=tone)
                    st.subheader("📄 Customized Proposal")
                    st.markdown(proposal)
                    st.info(f"🧠 OpenAI Token Estimate: {tokens} tokens")
                    st.download_button("Download Proposal (.txt)", proposal.encode(), file_name="proposal.txt")

                    st.subheader("🔍 Matched Case Studies")
                    for proj, score in similar:
                        st.markdown(f"**{proj['client_name']}** ({proj['industry']}) - Score: {score:.2f}")
                        st.caption(f"Tools: {', '.join(proj['tools']) if isinstance(proj['tools'], list) else proj['tools']}")
                        st.markdown(f"Problem: {proj['problem']}\n\nSolution: {proj['solution']}")
                else:
                    st.warning("No relevant case studies found. Please refine your input or add more data.")

    with tab2:
        with st.form("project_form"):
            st.subheader("Add New Project Knowledge")
            client_name = st.text_input("Client Name*")
            industry = st.text_input("Industry*")
            tools = st.text_input("Technologies Used (comma-separated)*")
            problem = st.text_area("Client Challenge*", height=100)
            solution = st.text_area("Our Solution*", height=100)
            outcome = st.text_area("Measured Outcomes*", height=100)
            project_link = st.text_input("Case Study Link*")
            portfolio_link = st.text_input("portfolio link*")

            if st.form_submit_button("Save Project"):
                if all([client_name, industry, tools, problem, solution, outcome, project_link, portfolio_link]):
                    new_project = {
                        "project_id": len(proposals) + 1,
                        "client_name": client_name,
                        "industry": industry,
                        "tools": [t.strip() for t in tools.split(",")],
                        "problem": problem,
                        "solution": solution,
                        "outcome": outcome,
                        "portfolio_link": portfolio_link,
                        "project_link": project_link
                    }
                    proposals.append(new_project)
                    save_proposals(proposals)
                    st.success("✅ Project added to knowledge base!")
                else:
                    st.error("Please fill all required fields (*)")

with col2:
    st.subheader("🧠 Project Knowledge Base")
    if proposals:
        for proj in proposals:
            with st.expander(f"{proj['client_name']} ({proj['industry']})"):
                st.markdown(f"""
                **Technologies:** {', '.join(proj['tools']) if isinstance(proj['tools'], list) else proj['tools']}  
                **Challenge:** {proj['problem']}  
                **Solution:** {proj['solution']}  
                **Outcome:** {proj['outcome']}  
                """)
                if proj['project_link']:
                    st.markdown(f"[Case Study]({proj['project_link']})")
    else:
        st.info("No projects in knowledge base. Add your first project!")
