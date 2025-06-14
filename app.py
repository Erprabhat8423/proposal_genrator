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
# st.write("Secrets loaded:", st.secrets)

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

# Proposal Generator

def generate_proposal(jd_text, similar):
    portfolio_link = similar[0][0]['portfolio_link'] if similar else ""
    case_study = "".join([
        f"""
    - Project: {proj['client_name']} ({proj['industry']})
      Problem: {proj.get('problem', '')}
      Our Solution: {proj.get('solution', '')}
      Results Achieved: {proj.get('outcome', '')}
      project link: {proj.get('project_link', '')}
    """
        for proj, _ in similar[:1]
    ]) if similar else ""

    prompt = f"""
    Generate a client proposal following this EXACT structure:

    - Start with: \"Dear Client,\"
    - Introduce yourself: \"My name is Tarsem Singh, and I am a Power BI expert with 8+ years of experience in business intelligence, data modeling, and dashboarding.\"
    - Briefly state your understanding of the client's needs.

    2. **Relevant Case Study** (Include ONLY if relevant)
    {case_study}

    3. **Proposed Approach**
    - Our understanding of your requirements
    - Step-by-step methodology
    - Technology stack recommendation
    - Questions for clarification (if needed)

    4. **Why Choose Me?**
    - Highlight your differentiators specific to the project needs
    - Include portfolio link: {portfolio_link}
    - Final call to action

    **Job Description:**
    {jd_text}

    **Mandatory Guidelines:**
    - Always start with \"Dear Client,\"
    - Always introduce yourself with \"My name is Tarsem Singh...\"
    - Always end with \"Best regards, Tarsem Singh\"
    - Use ONLY verified past projects from our database
    - Maximum 2 case studies, even 1 if highly relevant
    - Problem-Solution-Outcome format for case studies
    - Professional but conversational tone
    - Max 600 words
    - No technical jargon unless required

    **Strict Prohibitions:**
    - No fictional projects/statistics
    - No generic sales pitches
    - No assumptions beyond JD scope
    """
    # first_line = jd_text.strip().split("\n")[0] if jd_text.strip() else "your project requirements"

    # prompt = f"""
    # You are an expert proposal writer generating a simple, friendly, and custom proposal for a job description.

    # - Use simple, easy-to-understand language (8th–10th grade level).
    # - Do NOT use headings (no "Relevant Case Study", "Proposed Approach" etc.)
    # - Write as a natural flowing narrative, not in sections.
    # - Begin with "Dear Client,"
    # - Introduce yourself as: "My name is Tarsem Singh, and I am a Power BI expert with 8+ years of experience..."
    # - Show clear understanding of the client's needs.
    # - Briefly and naturally mention the case study below as a supporting example.
    # - Then explain how you will approach their needs (methodology, tools, tech stack).
    # - End with why you’re a good fit and include this portfolio link: {portfolio_link}
    # - Finish with "Best regards, Tarsem Singh"
    # - Keep the proposal under 450 words (max 600 tokens).
    # - Avoid technical jargon unless absolutely needed.

    # The client’s job description starts with: "{first_line}"

    # Here’s a relevant project you can refer to:
    # {case_study}

    # The full job description is:
    # {jd_text}
    # **Mandatory Guidelines:**
    # - Always start with \"Dear Client,\"
    # - Always introduce yourself with \"My name is Tarsem Singh...\"
    # - Always end with \"Best regards, Tarsem Singh\"
    # - Use ONLY verified past projects from our database
    # - Maximum 2 case studies, even 1 if highly relevant
    # - Problem-Solution-Outcome format for case studies
    # - Professional but conversational tone
    # - Max 450 words
    # - No technical jargon unless required

    # **Strict Prohibitions:**
    # - No fictional projects/statistics
    # - No generic sales pitches
    # - No assumptions beyond JD scope
    # """

    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(prompt))

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1000
    )

    return response.choices[0].message.content, token_count




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
                if similar:
                    proposal, tokens = generate_proposal(jd_text, similar)
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
