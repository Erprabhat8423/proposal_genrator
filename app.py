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

# Step 4: Generate Final Proposal
def generate_final_proposal(jd_summary, skills_text, hook_paragraph, jd_text, similar):
    portfolio_link = similar[0][0]['portfolio_link'] if similar else ""
    case_study = "\n\n".join([
        f"Project: {proj['client_name']} ({proj['industry']})\nProblem: {proj.get('problem', '')}\nOur Solution: {proj.get('solution', '')}\nResults Achieved: {proj.get('outcome', '')}\nProject Link: [View Project]({proj.get('project_link', '')})"
        for proj, _ in similar[:2]
    ]) if similar else ""

    prompt = f"""
    Write a professional, human-sounding proposal under 300 words using this structure:

    Paragraph 1 – Use this exact hook:
    {hook_paragraph}

    Paragraph 2 – Mention 1–2 relevant past projects and their impact. Include project links naturally.
    {case_study}
    If available, include links like: "You can explore that solution here: [View Project](...)"
    Paragraph 3 – Ask if the client has any preferences, data sources, or requirements.

    Paragraph 4 – Close with CTA and sign-off: "You can view more of my work here: {portfolio_link}\n\nBest regards,\nTarsem Singh"

    Guidelines:
    - Do not explain the job description.
    - Avoid salesy or robotic tone.
    - Use plain, conversational English.
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

# Step 4: Generate Final Proposal
# def generate_final_proposal(jd_summary, skills_text, hook_paragraph, jd_text, similar):
#     portfolio_link = similar[0][0]['portfolio_link'] if similar else ""
#     case_study = "\n\n".join([
#         f"""
#         - Project: {proj['client_name']} ({proj['industry']})
#         Problem: {proj.get('problem', '')}
#         Our Solution: {proj.get('solution', '')}
#         Results Achieved: {proj.get('outcome', '')}
#         Project Link: [View Project]({proj.get('project_link', '')})
#         """
#         for proj, _ in similar[:2]
#     ]) if similar else ""

#     jd_lower = jd_text.lower()
#     if "power bi" in jd_lower and "tableau" in jd_lower:
#         expert_intro = "My name is Tarsem Singh, and I’m a certified Power BI expert with over 8 years of experience. I’ve successfully delivered similar solutions using Power BI, Tableau, Informatica, and SAP BusinessObjects to help teams turn complex data into actionable insights."
#     elif "tableau" in jd_lower:
#         expert_intro = "My name is Tarsem Singh, and I’m a certified Tableau expert with over 8 years of experience. I’ve worked on various dashboard and BI solutions using Tableau and related tools to deliver impactful reporting and streamlined data workflows."
#     else:
#         expert_intro = "My name is Tarsem Singh, and I’m a certified Power BI expert with over 8 years of experience building dashboards, reporting systems, and end-to-end BI pipelines across diverse industries."

#     prompt = f"""
#     You are writing a professional proposal in simple, natural English that sounds like it came directly from a helpful and experienced freelancer — not an assistant.

#     Structure the proposal using the following format:

#     **Paragraph 1 – Introduction (Do NOT repeat the JD):**
#     Start with "Dear Client," 
#     Then introduce yourself as: "{expert_intro}"
#     Immediately follow with a 4–5 line custom hook that reflects the client's real needs and highlights relevant hard/soft skills based on the JD.

#     **Paragraph 2 – Relevant Projects (1-2 projects):**
#     Mention what the client needed, what you built, and what the impact was.
#     If available, include links like: "You can explore that solution here: [View Project](...)"

#     Use the following as reference:
#     {case_study}

#     **Paragraph 3 – Ask for Clarifications:**
#     Ask politely if the client has preferences, data sources, or specific requirements.

#     **Paragraph 4 – CTA + Sign-off:**
#     Finish with:
#     "I’d love to connect and discuss the project in more detail."
#     "You can view more of my work here: {portfolio_link}"
#     "Best regards,\nTarsem Singh"

#     Guidelines:
#     - Use simple, human language.
#     - No bullet points or headings in the final message.
#     - Keep it under 300 words.
#     - Write like a freelancer, not an AI assistant.

#     Client JD:
#     {jd_text}
#     """

#     encoding = tiktoken.encoding_for_model("gpt-4o")
#     token_count = len(encoding.encode(prompt))

#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.4,
#         max_tokens=800
#     )

#     return response.choices[0].message.content.strip(), token_count, prompt




# Proposal Generator
# def generate_proposal(jd_text, similar):
#     portfolio_link = similar[0][0]['portfolio_link'] if similar else ""
#     case_study = "".join([
#         f"""
#         - Project: {proj['client_name']} ({proj['industry']})
#         Problem: {proj.get('problem', '')}
#         Our Solution: {proj.get('solution', '')}
#         Results Achieved: {proj.get('outcome', '')}
#         Project Link: {proj.get('project_link', '')}
#         """
#         for proj, _ in similar[:1]
#     ]) if similar else ""


#     language_guidelines = """
#     - Use plain, simple English (8th to 10th grade reading level).
#     - Soften formal words: say “I’ve worked on…” instead of “I specialize in…”, and “I make sure…” instead of “I ensure…” insted of "we" use "I".
#     - Avoid resume-style language. Make it sound like a helpful expert having a conversation.
#     - Be warm, honest, and confident — not robotic or salesy.
#     """


#     first_line = jd_text.strip().split("\n")[0] if jd_text.strip() else "your project requirements"
#     # Detect tool preference from JD
#     jd_lower = jd_text.lower()
#     if "power bi" in jd_lower and "tableau" in jd_lower:
#         expert_intro = "My name is Tarsem Singh, and I am a Certified Power BI expert with 8+ years of hands-on development experience..."
#     elif "tableau" in jd_lower:
#         expert_intro = "My name is Tarsem Singh, and I am a Certified Tableau expert with 8+ years of hands-on development experience..."
#     else:
#         expert_intro = "My name is Tarsem Singh, and I am a Certified Power BI expert with 8+ years of hands-on development experience..."
#     prompt = f"""
#     You are an expert proposal writer generating a human-like, natural, and customized proposal based on the job description below.

#     - Write a flowing narrative that sounds human and consultative — not robotic.
#     - Start by introducing the applicant: "{expert_intro}"
#     - Immediately follow that with a 4–5 line custom hook that reflects the client's real needs and highlights relevant hard/soft skills based on the JD.
#     - For the last point: Tell what you've been doing or have experties with and how it relates to the client's needs later.
#     - Blend the hook naturally with the rest of the first paragraph — do not write it as a separate section or bullet points.
#     - Highlight a matching case study if available (use the format given below), and if a project link is present, include it naturally in the proposal using plain language like: "You can explore this case study here: [link]".
#     - Focus on how the developer will ensure clean architecture, collaboration with consultants, and meaningful reporting.
#      {language_guidelines}
#     - End with the portfolio link and signature: "Best regards, Tarsem Singh"

#     The client's job description starts with: "{first_line}"

#     Here’s a relevant project to reference:
#     {case_study}

#     The full job description is:
#     {jd_text}

#     **Must-Haves:**
#     - Start with "Dear Client," and introduce yourself immediately.
#     - Include a custom intro hook with key skills from the JD.
#     - Mention no more than 1–2 projects using Problem-Solution-Outcome.
#     - Use ONLY verified past projects from the data provided.
#     - End with: "You can view my portfolio here: {portfolio_link} \n\nBest regards, Tarsem Singh"
#     - Limit to 300 words.

#     **Prohibited:**
#     - No fake stats or stories.
#     - No generic phrases or templated filler.
#     - No headings or bullet points in the final proposal.
#     - No assumptions beyond the job description.
#     """

#     encoding = tiktoken.encoding_for_model("gpt-4o")
#     token_count = len(encoding.encode(prompt))

#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.5,
#         max_tokens=1000
#     )

#     return response.choices[0].message.content.strip(), token_count, prompt


# Streamlit UI
st.set_page_config(layout="wide")
col1, col2 = st.columns([8, 4])

with col1:
    st.title("🤖 AI Proposal Architect")
    tab1, tab2 = st.tabs(["📑 Generate Proposal", "📂 Add Project Data"])

    # with tab1:

    #     jd_text = st.text_area("Paste client's project requirements:", height=250)
    #     if st.button("Generate Proposal"):
    #         if jd_text.strip():
    #             similar = find_similar_proposals(jd_text)
    #             if similar:
    #                 proposal, tokens, prompt = generate_proposal(jd_text, similar)
    #                 st.subheader("📄 Customized Proposal")
    #                 st.markdown(proposal)
    #                 st.info(f"🧠 OpenAI Token Estimate: {tokens} tokens")
    #                 st.markdown("### Prompt Used:")
    #                 st.code(prompt, language="markdown")
    #                 st.download_button("Download Proposal (.txt)", proposal.encode(), file_name="proposal.txt")

    #                 st.subheader("🔍 Matched Case Studies")
    #                 for proj, score in similar:
    #                     st.markdown(f"**{proj['client_name']}** ({proj['industry']}) - Score: {score:.2f}")
    #                     st.caption(f"Tools: {', '.join(proj['tools']) if isinstance(proj['tools'], list) else proj['tools']}")
    #                     st.markdown(f"Problem: {proj['problem']}\n\nSolution: {proj['solution']}")
    #             else:
    #                 st.warning("No relevant case studies found. Please refine your input or add more data.")
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
