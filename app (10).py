import streamlit as st
import json
import os
import zipfile
import io
import shutil
import re
from PIL import Image  # For image processing
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import base64
import tiktoken
from langchain_groq import ChatGroq

GROQ_API_KEY = st.secrets["general"]["GROQ_API_KEY"]
st.set_page_config(page_title = 'ML/AI Research', page_icon = '📊', layout = 'wide')
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="qwen-qwq-32b")
MAX_TOKENS = 5000

# --- Constants and Config ---
PAPERS_PER_PAGE = 10
REQUIRED_DIRS = ['static/files', 'static/images', 'static/related_files', 'data']

# 
def setup_directories():
    for dir_path in REQUIRED_DIRS:
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists('data/papers.json'):
        with open('data/papers.json', 'w') as f:
            json.dump([], f)

# Load Papers Data
def load_papers():
    try:
        with open('data/papers.json', 'r') as f:
            papers = json.load(f)
            # Ensure each paper has a slug
            for paper in papers:
                if 'slug' not in paper:
                    paper['slug'] = generate_slug(paper['title'])
            return papers
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_papers(papers):
    with open('data/papers.json', 'w') as f:
        json.dump(papers, f, indent=2)

# sanitize filenames
def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()


# Database setup
def get_db_connection(): 
    try:
        conn=mysql.connector.connect(
            charset="utf8mb4",
            connection_timeout=10,
            database= st.secrets["general"]["database"],
            host="mysql-f3601b9-jonesjorney-bd4e.f.aivencloud.com",
            password=st.secrets["general"]["password"],
            port=21038,
            user=st.secrets["general"]["user"]
            )
        return conn
    
    except Error:
        st.error("Check Internet Connection and Try Again!")
        return []

def generate_slug(title):
    return re.sub(r'\W+', '-', title.lower())


#Zip file with related files for the paper
def create_zip(paper):
    zip_buffer = io.BytesIO()
    related_dir = os.path.join("static/related_files", paper.get('dir', ''))

    if os.path.exists(related_dir):
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(related_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname = file)

        zip_buffer.seek(0)
        st.download_button(
            label = "Download Related Files",
            data = zip_buffer,
            file_name = f"{paper['title']}_related_files.zip", mime = "application/zip"
        )            

def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def format_response(content: str) -> (str, str):
    if '<think>' in content and '</think>' in content:
        think = content.split('<think>')[1].split('</think>')[0].strip()
        resp = content.split('</think>',1)[1].strip()
    else:
        think, resp = None, content.strip()
    return think, resp


def get_contextual_response(user_input: str, paper_context: str) -> str:
    
    # Initialize messages if not exists
    session_key = f"ai_chat_{paper_context['slug']}"

    if session_key not in st.session_state:
        st.session_state[session_key] = [{
            "role": "system", 
            "content": f"""You are a research assistant. Use this context:
            Title: {paper_context['title']}
            Abstract: {paper_context.get('abstract','')}
            Objectives: {paper_context.get('objectives','')}
            Conclusion: {paper_context.get('conclusion','')}
            Summary: {paper_context.get('summary','')}
            Answer questions about this research."""
        }]

    # Add user message
    st.session_state[session_key].append({"role": "user", "content": user_input})

    # Truncate history
    token_count = count_tokens(st.session_state[session_key][0]['content'])
    truncated = [st.session_state[session_key][0]]
    
    for msg in reversed(st.session_state[session_key][1:]):
        t = count_tokens(msg['content'])
        if token_count + t > MAX_TOKENS: break
        truncated.insert(1, msg)
        token_count += t
    
    # Get response
    result = llm.invoke(truncated)
    think, resp = format_response(result.content)
    
    # Store response
    st.session_state[session_key].append({"role": "assistant", "content": resp})
    return think, resp


def clear_chat(slug):
    if f"messages_{slug}" in st.session_state:
        del st.session_state[f"messages_{slug}"]


def display_ai_chat(slug):
    papers = load_papers()
    paper = next((p for p in papers if p['slug'] == slug), None)
    
    if not paper:
        st.error("Paper not found")
        return
    
    st.subheader(f"Research Assistant: {paper['title']}")
    
    # Initialize session
    session_key = f"ai_chat_{slug}"
    if session_key not in st.session_state:
        st.session_state[session_key] = [{
            "role": "system", 
            "content": f"""You are a research assistant. Use this context:
            Title: {paper['title']}
            Abstract: {paper.get('abstract','')}
            Objectives: {paper.get('objectives','')}
            Conclusions: {paper.get('conclusions','')}
            EDA: {paper.get('eda','')}
            Answer questions about this research."""
        }]

    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state[session_key][1:]:  # Skip system message
            if msg['role'] == 'user':
                st.markdown(f"<div class='user-message' style='color: #000000;'>👤 {msg['content']}</div><div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-message' style='color: #000000;'>🤖 {msg['content']}</div><div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
    
    # Input
    # Input
    user_input = st.chat_input("Ask about this research...")
    if user_input:
        with st.spinner('Analyzing...'):
            think, resp = get_contextual_response(user_input, paper)
            # Update display
            st.rerun()
    
    # Clear chat button
     # Clear conversation button
    if st.button("Clear Chat"):
        del st.session_state[session_key]
        st.rerun()
    
    if st.button("← Return to Paper"):
        del st.query_params["chat"]
        st.rerun()
    
     # Add styling
    st.markdown("""
    <style>
    .user-message {background:#e3f2fd; padding:8px; border-radius:8px; max-width:80%; margin-left:20%;}
    .ai-message   {background:#f5f5f5; padding:8px; border-radius:8px; max-width:80%; margin-right:20%;}
    </style>
    """, unsafe_allow_html=True)



# Homepage
def display_home():
    
    image_column, text_column = st.columns([1, 3])

    with image_column:
        st.markdown("""
            <style>
                .circle-img {
                    border-radius: 50%;
                    overflow: hidden;
                    width: 200px;
                    height: 200px;
                    margin: auto;
                    position: relative;
                    top: 50%;
                    transform: translateY(25%);
                }
                .circle-img img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                /* Media query for mobile devices */
                @media screen and (max-width: 640px) {
                    .circle-img {
                        width: 150px;
                        height: 150px;
                        transform: translateY(0);
                        margin-bottom: 20px;
                    }
                }
            </style>
            <div class="circle-img">
                <img src="https://avatars.githubusercontent.com/u/169674746?s=400&u=98982bc9fafdfbc084b6426148a421fe35c80384&v=4" 
                    alt="Profile Picture">
            </div>
        """, unsafe_allow_html=True)
    with text_column:
        #st.title("Shivogo K. John")
        st.title("Machine Learning Eng. and Research")
        st.header("Supply Chains")
        st.markdown("""
            <div style="text-align: justify;">This platform offers a comprehensive collection of research and frameworks for modern supply chain management, focusing on enhancing resilience and operational efficiency. Key areas include AI applications in supplier risk management, predictive demand forecasting, and strategies to mitigate disruptions in supply chain operations. It provides valuable resources like methodologies, case studies, AI models, and sector-specific insights for logistics practitioners and researchers. The aim is to develop smarter, more resilient supply chains that leverage data for informed decision-making.</div>
        """, unsafe_allow_html=True)
        
        st.write("######")
        # Icons and links
        st.markdown("""
        <a href=""><img src="https://img.shields.io/badge/GitHub-000?style=flat&logo=github" alt="GitHub"></a>
        <a href=""><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin" alt="LinkedIn"></a>
        <a href="mailto:"><img src="https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail" alt="Email"></a>
        <a href=""><img src="https://img.shields.io/badge/Phone-25D366?style=flat&logo=whatsapp" alt="Phone"></a>
        """, unsafe_allow_html=True)

    # Add a subtle line separator
    st.markdown("""
                <div style="border-bottom: 1px solid #ccc; margin: 10px 0;"></div>
            """, unsafe_allow_html=True)
    st.write("####")

    # Display papers
    papers = load_papers()
    for idx, paper in enumerate(papers):
        col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 2, 2, 2])

        with col1:
            try:
                img = Image.open(paper['thumb_url'] ) #width = 1400
                img = img.resize((500, 450))
                st.image(img, use_container_width=True)
            except Exception:
                st.image(paper['thumb_url'], use_container_width=True)
        
        with col2:
            #st.markdown('<div style="min-height: 0px; padding-top: 0px;">', unsafe_allow_html=True)
            title = paper['title']
            if len(title) > 39:
                title = title[:39] + "..."
                st.markdown(f"""
                    <div title="{paper['title']}">
                        <h3 style="margin: 0;">{title}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='margin: 0;'>{title}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div style="padding-top: 30px;">', unsafe_allow_html=True)
            if st.button("Read PDF", key=f"read_{idx}"):
                st.query_params["read"] = paper["slug"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div style="padding-top: 30px;">', unsafe_allow_html=True)
            if st.button("🤖Ask AI ", key=f"ask_ai_{idx}"):
                st.query_params["chat"] = paper["slug"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div style="padding-top: 30px;">', unsafe_allow_html=True)
            with open(f"static/files/{paper['filename']}", "rb") as pdf_file:
                st.download_button(
                    label = 'Download PDF',
                    data = pdf_file,
                    file_name = paper['filename'],
                    key = f"dl_{idx}"
                )
            st.markdown('</div>', unsafe_allow_html=True)

            if paper.get('model_link'):
                st.markdown(f"""
                    <a href="{paper['model_link']}" target="_blank">
                        <button style="
                            background-color:#de7006;
                            color:black;
                            border:none;
                            padding:6px 12px;
                            border-radius:4px;
                            margin-top:30px;
                            cursor:pointer;
                        ">View Model</button>
                    </a>
                """, unsafe_allow_html=True)

            
 
        with col5:
            st.markdown('<div style="padding-top: 30px;">', unsafe_allow_html=True)
            if st.button("Related Files", key = f"rel_{idx}"):
                create_zip(paper)
            st.markdown('</div>', unsafe_allow_html=True)

        with col6:
            st.markdown('<div style="padding-top: 30px;">', unsafe_allow_html=True)
            if 'web_link' in paper and paper['web_link']:
                st.markdown(f"""
                    <a href="{paper['web_link']}" target="_blank">
                        <button style="
                            background-color: #de7006;
                            border: none;
                            color: black;
                            padding: 10px 20px;
                            cursor: pointer;
                            border-radius: 4px;
                        ">
                            <i class="fas fa-globe" style="color: black;"></i>
                        </button>
                    </a>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
        st.write("---")

        # CSS for curved thumbnail edges
        st.markdown("""
            <style>
            img {
                border-radius: 12px;
            }
            </style>
        """, unsafe_allow_html=True)

def admin_login():
        
    # Admin Credentials
    ADMIN_USERNAME = st.secrets["general"]["ADMIN_USERNAME"]
    ADMIN_PASSWORD = st.secrets["general"]["ADMIN_PASSWORD"]

    st.title("Admin Login")
    if st.session_state.logged_in:
        admin_panel()

        if st.button("Logout"):
            st.session_state.logged_in = False
            #st.experimental_rerun()
            st.session_state.active_section = "Admin Login" 
        return

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type = "password")

        if st.form_submit_button("Login"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.active_section = "Admin Panel" 
                #st.experimental_rerun()
            else:
                st.error("Invalid credentials")  

def admin_panel():
    st.title("Admin Panel")
    
    with st.form("upload_form"):
        title = st.text_input("Paper Title")
        date = st.date_input("Upload Date")
        web_link = st.text_input("Web Link (Optional)") # Added web_link field
        model_link = st.text_input("Model Link (URL)")
        pdf_file = st.file_uploader("PDF File", type = ["pdf"])
        thumb_file = st.file_uploader("Thumbnail", type = ["png", "jpg", "jpeg"])
        related_files = st.file_uploader("Related Files", accept_multiple_files = True)
        abstract = st.text_area("Abstract")
        conclusion = st.text_area("Conclusion")
        objectives = st.text_area("Research Objectives")
        summary = st.text_area("Summary")
        
        if st.form_submit_button("Upload Paper"):
            if all([title,date, pdf_file, thumb_file]):
                #save main files
                pdf_filename = pdf_file.name
                thumb_filename = thumb_file.name

                #save pdf file
                with open(f"static/files/{pdf_filename}", "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                # save thumbnail
                with open(f"static/images/{thumb_filename}", "wb") as f:
                    f.write(thumb_file.getbuffer())
                
                # Handle related files
                paper_dir = os.path.splitext(pdf_filename)[0]
                related_dir = os.path.join("static/related_files", paper_dir)

                os.makedirs(related_dir, exist_ok =  True)

                related_filenames = []

                for file in related_files:
                    with open(os.path.join(related_dir, file.name), "wb") as f:
                        f.write(file.getbuffer())
                    related_filenames.append(file.name)

                slug = generate_slug(title)
                # Update metadata
                new_paper ={
                    "title" : title,
                    "slug": slug,
                    "filename" : pdf_filename,
                    "thumb_url" : f"static/images/{thumb_filename}",
                    "title" : title,
                    "upload_date" : str(date),
                    "dir": paper_dir,
                    "related_files": related_filenames,
                    "web_link": web_link if web_link else "" , # Added web_link to paper metadata
                    "model_link": model_link if model_link else "", 
                    "abstract": abstract,
                    "conclusion": conclusion,
                    "objectives": objectives,
                    "summary": summary,
                }
                papers = load_papers()
                papers.insert(0, new_paper)
                save_papers(papers)
                st.success("Paper uploaded successfully!")
            else:
                st.error("Please fill all required fields")

    # Paper Management
    papers = load_papers()
    for idx, paper in enumerate(papers):
        cols = st.columns([9,1])
        cols[0].subheader(paper['title'])

        button_key = f"del-button-{paper['title']}-{idx}-{paper['filename']}"
        if cols[1].button("Delete", key=button_key):
            delete_paper(paper)
            st.rerun()

def delete_paper(paper):
    try:
        # Remove files
        os.remove(f"static/files/{paper['filename']}")
        os.remove(f"static/images/{paper['thumb_url'].split('/')[-1]}")
        shutil.rmtree(f"static/related_files/{paper['dir']}")
        
        # Update metadata
        papers = load_papers()
        papers = [p for p in papers if p['title'] != paper['title']]
        save_papers(papers)
    except Exception as e:
        st.error(f"Deletion failed: {str(e)}")


def display_contact():
    st.title("Contact")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")

        if st.form_submit_button("Send"):
            conn = get_db_connection()
            if conn is None:
                flash("Failed to connect to the database.", "danger")

            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("INSERT INTO messages(name, email, content) VALUES(%s, %s, %s)", (name, email, message))
                conn.commit()
            except Error:
                st.error("Check Internet Connection and Try Again!")
                return []
            finally:
                conn.close()
            st.success("Message sent successfully!")  
            st.success("Await a response!")  



def scroll_to_top():
    st.markdown("""
        <div style="position: fixed; bottom: 10px; right: 10px;">
            <a href="##supply-chains">
                <button style="
                    background: #000000;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 50%;
                    cursor: pointer;
                ">▲</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

#Css styling for buttons
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color:#de7006;  /* Button background color */
        color: #000000;  /* Button text color */
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #de7006;  /* No change on hover */
        border-color: none;      /* No change on hover */
        color: #000000;             /* No change on hover */
    }
    div.stButton > button:active {
        background-color: #de7006;  /* No change on click */
        border-color: none;      /* No change on click */
        color: #000000;             /* No change on click */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div.stDownloadButton > button {
        background-color:#de7006;  /* Button background color */
        color: #000000;  /* Button text color */
        cursor: pointer;
    }
    div.stDownloadButton > button:hover {
        background-color: #de7006;  /* No change on hover */
        border-color: none;      /* No change on hover */
        color: #000000;             /* No change on hover */
    }
    div.stDownloadButton > button:active {
        background-color: #de7006;  /* No change on click */
        border-color: none;      /* No change on click */
        color: #000000;             /* No change on click */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- UI Enhancements ---
def add_background():
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("https://raw.githubusercontent.com/SHIVOGOJOHN/Research-Paper-Tool/main/static/images/background9.jpg.jpg");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                -webkit-background-size: cover;
                -moz-background-size: cover;
                -o-background-size: cover;
            }}
            .main {{
                background: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 10px;
                backdrop-filter: blur(5px);
            }}
        </style>
    """, unsafe_allow_html=True)

##
def display_read_pdf(slug):
    papers = load_papers()
    paper = next((p for p in papers if p["slug"] == slug), None)
    
    if not paper:
        st.error("Paper not found.")
        return

    #st.title(f"Reading: {paper['title']}")
    
    pdf_path = f"static/files/{paper['filename']}"
    
    try:
        # Always provide a direct download option
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
            # First try embedded display
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f"""
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" height="800px" type="application/pdf">
                </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Provide download button as backup
            st.download_button(
                "Download PDF",
                pdf_bytes,
                file_name=paper['filename'],
                mime="application/pdf",
                help="If the PDF doesn't display properly, download it to view"
            )
    except Exception as e:
        st.error("Error displaying PDF. Please use the download button below.")
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                f.read(),
                file_name=paper['filename'],
                mime="application/pdf"
            )
    
    if st.button("Return to Home"):
        del st.query_params["read"]
        st.rerun()


def main():
    add_background()
    setup_directories()
    # Admin Credentials
    
    ADMIN_USERNAME = st.secrets["general"]["ADMIN_USERNAME"]
    ADMIN_PASSWORD = st.secrets["general"]["ADMIN_PASSWORD"]
    
    st.sidebar.write("---")
    
    query_params = st.query_params
    # 🟢 Check if the user clicked a 'Read' link
    if "read" in query_params:
        slug = query_params["read"]
        display_read_pdf(slug)
        return

    #Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Handle AI chat routing
    if "chat" in query_params:
        display_ai_chat(query_params["chat"])
        return
        
    options = ['Home', 'Admin Login','Contact']
    #Navigation
    page = st.sidebar.selectbox("**Menu**", options)
    st.sidebar.write("---")

    papers = load_papers()
    if page == "Home":
        scroll_to_top()
        display_home()
    elif page == "Admin Login":
        admin_login()
    elif page == "Contact":
        display_contact()

if __name__ == "__main__":
    main()   
