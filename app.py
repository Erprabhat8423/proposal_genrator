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

def summarize_jd(jd_text):
    prompt = f"""
    Summarize the following job description in 3–5 lines, focusing on the client's core need, expected deliverables, and tools mentioned.
    Job Description:
    {jd_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# Step 2: Extract Skills
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
    return response.choices[0].message.content.strip()

# Step 3: Generate Hook Paragraph
def generate_hook_paragraph(jd_text):
    prompt = f"""
    From the job description below, generate a short paragraph (max 5 lines) that:
    - Starts with "Dear Client," then follows with a confident introduction.
    - States: "My name is Tarsem Singh, and I’m a certified [Tool] expert with over 8 years of experience."
    - Then immediately says you've worked with the same tools and business problems in the JD.
    - Do not repeat the job description — instead, say how you’ve already done similar things.
    - Use simple, confident, natural English — like a freelancer speaking directly.

    Job Description:
    {jd_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def extract_clarification_questions(jd_text, max_q=3):
    """
    Use GPT-4o to pull out concrete clarifying questions the freelancer
    should ask the client. Returns a list of short questions (strings).
    """
    prompt = f"""
    From the job description below, list up to {max_q} concrete questions
    you would ask the client to remove any ambiguity (timeline, data access, 
    success metrics, tech stack constraints, etc.). 
    • Only include questions whose answers are NOT obvious in the text. 
    • Return each question on a new line, no numbering, no extra text.

    Job Description:
    {jd_text}
    """
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )
    # Split on newlines and remove empties
    return [q.strip() for q in resp.choices[0].message.content.split("\n") if q.strip()]


# Step 4: Generate Final Proposal
def generate_final_proposal(jd_summary, skills_text, hook_paragraph, jd_text, similar):
    portfolio_link = similar[0][0]['portfolio_link'] if similar else ""
    case_study = "\n\n".join([
        f"Project: {proj['client_name']} ({proj['industry']})\nProblem: {proj.get('problem', '')}\nOur Solution: {proj.get('solution', '')}\nResults Achieved: {proj.get('outcome', '')}\nProject Link: [View Project]({proj.get('project_link', '')})"
        for proj, _ in similar[:2]
    ]) if similar else ""

    clar_qs = extract_clarification_questions(jd_text)
    if clar_qs:
        clar_section = "Here are a few quick clarifications to ensure we hit the ground running:\n" + \
                       "\n".join(f"- {q}" for q in clar_qs)
    else:
        clar_section = "Do you have any preferences, data sources, or special constraints we should be aware of?"

    prompt = f"""
    Write a professional, human-sounding proposal under 300 words using this structure:

    Paragraph 1 – Use this exact hook:
    {hook_paragraph}

    Paragraph 2 – Mention 1–2 relevant past projects and their impact. Include project links naturally.
    {case_study}
    If available, include links like: "You can explore that solution here: [View Project](...)"
    Paragraph 3 – Ask clarifying questions if needed.
    {clar_section}

    Paragraph 4 – Close with CTA and sign-off: "You can view more of my work here: {portfolio_link}\n\nBest regards,\nTarsem Singh"

    Guidelines:
    - Do not explain the job description.
    - Avoid salesy or robotic tone.
    - Use plain, conversational English.
    - Always start with \"Dear Client,\"
    - Always introduce yourself with \"My name is Tarsem Singh...\"
    - Always end with \"Best regards, Tarsem Singh\"
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    token_count = len(encoding.encode(prompt))

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800
    )

    return response.choices[0].message.content.strip(), token_count, prompt


# Streamlit UI
st.set_page_config(layout="wide")
col1, col2 = st.columns([8, 4])

with col1:
    st.title("🤖 AI Proposal Architect")
    tab1, tab2 = st.tabs(["📑 Generate Proposal", "📂 Add Project Data"])

    with tab1:
        jd_text = st.text_area("Paste client's project requirements:", height=250)
        if st.button("Generate Proposal"):
            if jd_text.strip():
                similar = find_similar_proposals(jd_text)
                jd_summary = summarize_jd(jd_text)
                skills_text = extract_skills_with_llm(jd_text)
                hook_paragraph = generate_hook_paragraph(jd_text)
                proposal, tokens, prompt = generate_final_proposal(jd_summary, skills_text, hook_paragraph, jd_text, similar)

                st.subheader("📄 Customized Proposal")
                st.markdown(proposal)
                st.info(f"🧠 OpenAI Token Estimate: {tokens} tokens")
                st.download_button("Download Proposal (.txt)", proposal.encode(), file_name="proposal.txt")

                with st.expander("🧠 Prompt Used"):
                    st.code(prompt, language="markdown")

                st.subheader("🔍 Matched Case Studies")
                for proj, score in similar:
                    st.markdown(f"**{proj['client_name']}** ({proj['industry']}) - Score: {score:.2f}")
                    st.caption(f"Tools: {', '.join(proj['tools']) if isinstance(proj['tools'], list) else proj['tools']}")
                    st.markdown(f"Problem: {proj['problem']}\n\nSolution: {proj['solution']}")

                with st.expander("🧠 Extracted Job Summary"):
                    st.markdown(jd_summary)

                with st.expander("🛠️ Extracted Skills"):
                    st.markdown(skills_text)

                with st.expander("🎯 Custom Hook"):
                    st.markdown(hook_paragraph)
            else:
                st.warning("Please paste a job description before generating.")
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
