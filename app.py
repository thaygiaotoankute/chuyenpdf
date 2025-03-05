import streamlit as st
import time
import requests
import zipfile
import os
import re
import tempfile
import shutil
import hashlib
import json
from datetime import datetime
import io
import base64
import PyPDF2
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(
    page_title="P_Convert_2025_Special (PDF:Word-Image-Equation)",
    page_icon="📄",
    layout="wide"
)

# Khởi tạo session state cho authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'hardware_id' not in st.session_state:
    st.session_state.hardware_id = ""
if 'activation_status' not in st.session_state:
    st.session_state.activation_status = "CHƯA KÍCH HOẠT"

# URL của file users.json trên GitHub (raw content)
USERS_FILE_URL = "https://raw.githubusercontent.com/thayphuctoan/pconvert/refs/heads/main/user.json"
# URL của file activated.txt trên GitHub (raw content)
ACTIVATION_FILE_URL = "https://raw.githubusercontent.com/thayphuctoan/pconvert/main/check-convert"

# Hàm lấy danh sách người dùng từ GitHub
@st.cache_data(ttl=300)  # Cache 5 phút
def get_users():
    try:
        response = requests.get(USERS_FILE_URL)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            st.error(f"Không thể lấy danh sách người dùng từ GitHub. Status code: {response.status_code}")
            return {}
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách người dùng: {str(e)}")
        return {}

# Hàm lấy danh sách ID đã kích hoạt từ GitHub
@st.cache_data(ttl=300)  # Cache 5 phút
def get_activated_ids():
    try:
        response = requests.get(ACTIVATION_FILE_URL)
        if response.status_code == 200:
            return response.text.strip().split('\n')
        else:
            st.error(f"Không thể lấy danh sách ID kích hoạt từ GitHub. Status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách ID kích hoạt: {str(e)}")
        return []

# Hàm xác thực người dùng
def authenticate_user(username, password):
    users = get_users()
    if username in users and users[username] == password:
        return True
    return False

# Hàm tạo hardware ID cố định từ username
def generate_hardware_id(username):
    # Tạo hardware ID từ username - luôn giống nhau cho cùng một username
    hardware_id = hashlib.md5(username.encode()).hexdigest().upper()
    formatted_id = '-'.join([hardware_id[i:i+8] for i in range(0, len(hardware_id), 8)])
    return formatted_id + "-Premium"

# Hàm kiểm tra kích hoạt
def check_activation(hardware_id):
    activated_ids = get_activated_ids()
    
    if hardware_id in activated_ids:
        return True
    else:
        return False

# Trang đăng nhập
def login_page():
    st.title("P_Convert - Đăng nhập")
    
    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")
    
    if st.button("Đăng nhập"):
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            
            # Tạo và lưu hardware ID
            hardware_id = generate_hardware_id(username)
            st.session_state.hardware_id = hardware_id
            
            # Kiểm tra trạng thái kích hoạt
            is_activated = check_activation(hardware_id)
            st.session_state.activation_status = "ĐÃ KÍCH HOẠT" if is_activated else "CHƯA KÍCH HOẠT"
            
            # Hiển thị hardware ID để thêm vào danh sách kích hoạt
            st.success(f"Đăng nhập thành công! Hardware ID của bạn là: {hardware_id}")
            
            if is_activated:
                st.success("Tài khoản của bạn đã được kích hoạt!")
                if st.button("Tiếp tục"):
                    st.experimental_rerun()
            else:
                st.warning("Tài khoản của bạn chưa được kích hoạt!")
                st.info("Vui lòng liên hệ quản trị viên để kích hoạt hardware ID này.")
                if st.button("Thử lại sau"):
                    st.experimental_rerun()
        else:
            st.error("Tên đăng nhập hoặc mật khẩu không đúng!")
    
    st.write("Chưa có tài khoản? Vui lòng liên hệ quản trị viên để được cấp.")

# ------------------------------ Utility Functions ------------------------------
def split_pdf(input_pdf_data, pages_per_part=5):
    """Split a PDF into multiple parts"""
    parts = []
    reader = PyPDF2.PdfReader(io.BytesIO(input_pdf_data))
    total_pages = len(reader.pages)
    
    for start in range(0, total_pages, pages_per_part):
        writer = PyPDF2.PdfWriter()
        for i in range(start, min(start + pages_per_part, total_pages)):
            writer.add_page(reader.pages[i])
        
        # Create bytes buffer for the split PDF
        output_buffer = io.BytesIO()
        writer.write(output_buffer)
        output_buffer.seek(0)
        
        parts.append({
            "name": f"Part {start//pages_per_part + 1} (Pages {start+1}-{min(start+pages_per_part, total_pages)})",
            "data": output_buffer.getvalue()
        })
    
    return parts

def get_timeout(file_size=None):
    """Get appropriate timeout values based on file size"""
    if file_size is not None:
        if file_size < 5 * 1024 * 1024:  # 5MB
            return (10, 30)
    return (10, 180)

def download_and_read_full_md(zip_url):
    """Download and extract markdown content from zip file"""
    try:
        # Create temp directory to extract files
        temp_dir = tempfile.mkdtemp()
        
        # Download the zip file
        timeout_val = get_timeout()
        resp = requests.get(zip_url, timeout=timeout_val)
        
        if resp.status_code != 200:
            return f"Lỗi: Tải ZIP thất bại. HTTP {resp.status_code}"
        
        # Save the zip file
        zip_path = os.path.join(temp_dir, "output.zip")
        with open(zip_path, 'wb') as f:
            f.write(resp.content)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            md_file_name = None
            for f_name in zip_ref.namelist():
                if f_name.endswith(".md"):
                    md_file_name = f_name
                    break
            
            if not md_file_name:
                shutil.rmtree(temp_dir)
                return "Lỗi: Không tìm thấy file .md trong gói kết quả!"
            
            zip_ref.extractall(temp_dir)
        
        # Read the markdown file
        md_path = os.path.join(temp_dir, md_file_name)
        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Process images in the markdown
        images_dir = os.path.join(temp_dir, 'images')
        if os.path.exists(images_dir):
            md_text = update_md_image_paths(md_text, images_dir)
        
        # Clean up the temp directory
        shutil.rmtree(temp_dir)
        
        return md_text
    
    except Exception as e:
        return f"Lỗi: Không thể xử lý file ZIP: {str(e)}"

def update_md_image_paths(md_text, image_folder_path):
    """Update image paths in markdown text"""
    def replace_path(match):
        img_path = match.group(2)
        if img_path.startswith('images/'):
            image_name = os.path.basename(img_path)
            img_file_path = os.path.join(image_folder_path, image_name)
            
            if os.path.exists(img_file_path):
                # Convert image to base64 for embedding
                with open(img_file_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                # Get image format from filename
                img_format = os.path.splitext(image_name)[1][1:].lower()
                if img_format == 'jpg':
                    img_format = 'jpeg'
                
                return f'![{match.group(1)}](data:image/{img_format};base64,{img_data})'
        
        return match.group(0)
    
    pattern = r'!\[(.*?)\]\((images/[^)]+)\)'
    return re.sub(pattern, replace_path, md_text)

def convert_tables_to_md(text):
    """Convert HTML tables to markdown format"""
    def html_table_to_markdown(table_tag):
        rows = table_tag.find_all('tr')
        max_cols = 0
        for row in rows:
            count_cols = 0
            for cell in row.find_all(['td', 'th']):
                colspan = int(cell.get('colspan', 1))
                count_cols += colspan
            max_cols = max(max_cols, count_cols)
        
        grid = [["" for _ in range(max_cols)] for _ in range(len(rows))]
        
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                while col_idx < max_cols and grid[row_idx][col_idx] != "":
                    col_idx += 1
                
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                cell_text = cell.get_text(strip=True)
                
                grid[row_idx][col_idx] = cell_text
                
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r == row_idx and c == col_idx:
                            continue
                        grid[r][c] = ""
                
                col_idx += colspan
        
        md_lines = []
        header_rows = 1
        
        for hr in range(header_rows):
            md_lines.append("| " + " | ".join(grid[hr]) + " |")
        
        align_line = "| " + " | ".join(["---"] * max_cols) + " |"
        md_lines.insert(header_rows, align_line)
        
        for row_idx in range(header_rows, len(rows)):
            md_lines.append("| " + " | ".join(grid[row_idx]) + " |")
        
        return "\n".join(md_lines)
    
    if '<html>' in text and '<table' in text:
        html_parts = text.split('</html>')
        final_text = text
        
        for part in html_parts:
            if '<html>' in part and '<table' in part:
                html_chunk = part[part.find('<html>'):]
                if not html_chunk.endswith('</html>'):
                    html_chunk += '</html>'
                
                soup = BeautifulSoup(html_chunk, 'html.parser')
                tables = soup.find_all('table')
                
                for table in tables:
                    md_table = html_table_to_markdown(table)
                    final_text = final_text.replace(str(table), md_table, 1)
        
        return final_text
    
    return text

def call_gemini_api(original_text, gemini_key):
    """Call Gemini API to correct Vietnamese spelling and grammar"""
    try:
        if not gemini_key:
            return "Lỗi: Chưa có Gemini API Key"
        
        GEMINI_API_URL = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash-002:generateContent?key=" + gemini_key
        )
        
        prompt = (
            "Please help me correct Vietnamese spelling and grammar in the following text. "
            "IMPORTANT: Do not change any image paths, LaTeX formulas, or Vietnamese diacritical marks. "
            "Return only the corrected text with the same structure and markdown formatting:\n\n"
            f"{original_text}"
        )
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
            }
        }
        
        headers = {"Content-Type": "application/json"}
        resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=(10, 180))
        
        if resp.status_code == 200:
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    corrected_text = candidate["content"]["parts"][0].get("text", "")
                    if corrected_text.strip():
                        return corrected_text
            
            return "Lỗi: Không thể trích xuất được kết quả từ Gemini API."
        else:
            return f"Lỗi: Gemini API - HTTP {resp.status_code} - {resp.text}"
    
    except Exception as e:
        return f"Lỗi: Gọi Gemini API thất bại: {e}"

def upload_and_parse_pdf(pdf_data, mineru_token, progress_callback=None):
    """Upload and parse PDF using Mineru API"""
    try:
        # Step 1: Get upload URL
        url_batch = "https://mineru.net/api/v4/file-urls/batch"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {mineru_token}"}
        data = {
            "enable_formula": True,
            "enable_table": True,
            "layout_model": "doclayout_yolo",
            "language": "vi",
            "files": [{"name": "demo.pdf", "is_ocr": True, "data_id": "abcd1234"}]
        }
        
        timeout_val = get_timeout(len(pdf_data))
        
        if progress_callback:
            progress_callback(5)
        
        resp = requests.post(url_batch, headers=headers, json=data, timeout=timeout_val)
        
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}: {resp.text}"
        
        rj = resp.json()
        code = rj.get("code")
        
        if code not in [0, 200]:
            return f"Mã lỗi: {code}, msg: {rj.get('msg')}"
        
        batch_id = rj["data"]["batch_id"]
        file_urls = rj["data"]["file_urls"]
        
        if not file_urls:
            return "Không có link upload trả về."
        
        upload_url = file_urls[0]
        
        if progress_callback:
            progress_callback(10)
        
        # Step 2: Upload PDF
        up_resp = requests.put(upload_url, data=pdf_data, timeout=timeout_val)
        
        if up_resp.status_code != 200:
            return f"Upload thất bại, HTTP {up_resp.status_code}"
        
        if progress_callback:
            progress_callback(20)
        
        # Step 3: Poll for results
        url_get = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        headers_poll = {"Content-Type": "application/json", "Authorization": f"Bearer {mineru_token}"}
        timeout_val_poll = get_timeout(len(pdf_data))
        
        max_retry = 30
        for i in range(max_retry):
            time.sleep(5)
            progress = 20 + int(((i+1)/max_retry) * 70)
            
            if progress_callback:
                progress_callback(progress)
            
            r = requests.get(url_get, headers=headers_poll, timeout=timeout_val_poll)
            
            if r.status_code == 200:
                rj = r.json()
                code = rj.get("code")
                
                if code in [0, 200]:
                    extract_result = rj["data"].get("extract_result", [])
                    
                    if extract_result:
                        res = extract_result[0]
                        state = res.get("state", "")
                        
                        if state == "done":
                            full_zip_url = res.get("full_zip_url", "")
                            
                            if not full_zip_url:
                                return "Không tìm thấy link kết quả!"
                            
                            md_text = download_and_read_full_md(full_zip_url)
                            
                            if md_text.startswith("Lỗi:"):
                                return md_text
                            
                            if progress_callback:
                                progress_callback(100)
                            
                            return md_text
                        
                        elif state == "failed":
                            err_msg = res.get("err_msg", "Unknown error")
                            return f"Task failed: {err_msg}"
            else:
                return f"Poll thất bại HTTP {r.status_code}"
        
        return "Hết thời gian chờ. Vui lòng thử lại sau."
    
    except Exception as e:
        return f"Lỗi: {str(e)}"

def convert_md_to_docx(md_content):
    """
    Convert markdown to Word document
    Note: This is a placeholder. In a real Streamlit app, we'd use a different approach
    since we can't use local Pandoc installation
    """
    # For now, we'll use a simple approach to create a downloadable Word file
    try:
        from docx import Document
        
        doc = Document()
        doc.add_paragraph(md_content)
        
        # Save to a BytesIO object
        docx_bytes = io.BytesIO()
        doc.save(docx_bytes)
        docx_bytes.seek(0)
        
        return docx_bytes.getvalue()
    except Exception as e:
        st.error(f"Error creating Word document: {str(e)}")
        return None

def create_download_link(data, filename, text):
    """Create a download link for a file"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ------------------------------ Session State Setup ------------------------------
def init_session_state():
    """Initialize session state variables"""
    if 'mineru_token' not in st.session_state:
        st.session_state.mineru_token = ""
    
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None
    
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = ""
    
    if 'result_text' not in st.session_state:
        st.session_state.result_text = ""
    
    if 'split_parts' not in st.session_state:
        st.session_state.split_parts = []
        
    # These authentication-related states are already defined at the top
    # but we add them here too for completeness
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = ""
        
    if 'hardware_id' not in st.session_state:
        st.session_state.hardware_id = ""
        
    if 'activation_status' not in st.session_state:
        st.session_state.activation_status = "CHƯA KÍCH HOẠT"

# ------------------------------ Main App ------------------------------
def main():
    # Initialize session state
    init_session_state()
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
        return
        
    # Check if user is activated
    if st.session_state.activation_status != "ĐÃ KÍCH HOẠT":
        st.title("P_Convert - Trạng thái kích hoạt")
        st.warning("Tài khoản của bạn chưa được kích hoạt!")
        st.info(f"Hardware ID: {st.session_state.hardware_id}")
        st.info("Vui lòng liên hệ quản trị viên để kích hoạt hardware ID này.")
        
        if st.button("Kiểm tra lại"):
            # Re-check activation status
            is_activated = check_activation(st.session_state.hardware_id)
            st.session_state.activation_status = "ĐÃ KÍCH HOẠT" if is_activated else "CHƯA KÍCH HOẠT"
            st.experimental_rerun()
            
        if st.button("Đăng xuất"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.hardware_id = ""
            st.session_state.activation_status = "CHƯA KÍCH HOẠT"
            st.experimental_rerun()
        return
    
    # App header
    st.title("P_Convert_2025_Special (PDF:Word-Image-Equation)")
    st.write("Công cụ chuyển đổi PDF thành văn bản, hình ảnh và công thức toán học")
    
    # Show user info
    st.sidebar.success(f"Đăng nhập: {st.session_state.username}")
    st.sidebar.info(f"Hardware ID: {st.session_state.hardware_id}")
    st.sidebar.success(f"Trạng thái: {st.session_state.activation_status}")
    
    if st.sidebar.button("Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.hardware_id = ""
        st.session_state.activation_status = "CHƯA KÍCH HOẠT"
        st.experimental_rerun()
    
    # API Keys section
    with st.expander("API Keys", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            mineru_token = st.text_input(
                "Mineru Token",
                type="password",
                value=st.session_state.mineru_token,
                key="mineru_input"
            )
            st.session_state.mineru_token = mineru_token
        
        with col2:
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                key="gemini_input"
            )
            st.session_state.gemini_api_key = gemini_api_key
    
    # File Upload Section
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"], key="pdf_uploader")
    
    if uploaded_file is not None:
        # Store PDF data in session state
        st.session_state.pdf_data = uploaded_file.getvalue()
        st.session_state.pdf_name = uploaded_file.name
        
        st.success(f"Đã tải lên: {uploaded_file.name}")
        
        # Create tabs for processing options
        tab1, tab2 = st.tabs(["Xử lý toàn bộ PDF", "Tách và xử lý từng phần"])
        
        # Tab 1: Process entire PDF
        with tab1:
            if st.button("Upload & Parse (Toàn bộ PDF)", key="process_all"):
                if not st.session_state.mineru_token:
                    st.error("Vui lòng nhập Mineru Token trước khi upload.")
                else:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Đang upload và phân tích PDF...")
                    
                    # Process the PDF
                    result = upload_and_parse_pdf(
                        st.session_state.pdf_data,
                        st.session_state.mineru_token,
                        progress_callback=lambda p: progress_bar.progress(p/100)
                    )
                    
                    if result.startswith("Lỗi") or result.startswith("HTTP") or result.startswith("Mã lỗi"):
                        st.error(result)
                    else:
                        status_text.text("Đang hiệu đính với Gemini API...")
                        
                        # Call Gemini for correction if API key is provided
                        if st.session_state.gemini_api_key:
                            corrected_text = call_gemini_api(result, st.session_state.gemini_api_key)
                            
                            if corrected_text.startswith("Lỗi"):
                                st.warning(corrected_text)
                                final_text = result
                            else:
                                final_text = corrected_text
                        else:
                            final_text = result
                        
                        # Store the result
                        st.session_state.result_text = final_text
                        status_text.text("Hoàn thành!")
                        
                        # Show the result
                        st.markdown("### Kết quả")
                        st.markdown(final_text)
        
        # Tab 2: Split and process parts
        with tab2:
            if st.button("Tách PDF", key="split_pdf"):
                # Split the PDF
                with st.spinner("Đang tách PDF..."):
                    st.session_state.split_parts = split_pdf(st.session_state.pdf_data)
                
                st.success(f"Đã tách thành {len(st.session_state.split_parts)} phần")
            
            # Display split parts if available
            if st.session_state.split_parts:
                st.subheader("Các phần đã tách")
                
                # Create selection for parts
                part_names = [part["name"] for part in st.session_state.split_parts]
                selected_part = st.selectbox("Chọn phần để xử lý", part_names)
                
                # Get index of selected part
                selected_index = part_names.index(selected_part)
                
                # Process selected part
                if st.button("Xử lý phần đã chọn", key="process_part"):
                    if not st.session_state.mineru_token:
                        st.error("Vui lòng nhập Mineru Token trước khi upload.")
                    else:
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Đang upload và phân tích phần đã chọn...")
                        
                        # Process the selected part
                        selected_part_data = st.session_state.split_parts[selected_index]["data"]
                        result = upload_and_parse_pdf(
                            selected_part_data,
                            st.session_state.mineru_token,
                            progress_callback=lambda p: progress_bar.progress(p/100)
                        )
                        
                        if result.startswith("Lỗi") or result.startswith("HTTP") or result.startswith("Mã lỗi"):
                            st.error(result)
                        else:
                            status_text.text("Đang hiệu đính với Gemini API...")
                            
                            # Call Gemini for correction if API key is provided
                            if st.session_state.gemini_api_key:
                                corrected_text = call_gemini_api(result, st.session_state.gemini_api_key)
                                
                                if corrected_text.startswith("Lỗi"):
                                    st.warning(corrected_text)
                                    final_text = result
                                else:
                                    final_text = corrected_text
                            else:
                                final_text = result
                            
                            # Append to existing result
                            if st.session_state.result_text:
                                new_text = f"\n\n--- Kết quả từ {selected_part} ---\n{final_text}"
                                st.session_state.result_text += new_text
                            else:
                                st.session_state.result_text = f"--- Kết quả từ {selected_part} ---\n{final_text}"
                            
                            status_text.text("Hoàn thành!")
                            
                            # Show the result
                            st.markdown("### Kết quả")
                            st.markdown(st.session_state.result_text)
                
                # Process all parts
                if st.button("Xử lý tất cả các phần", key="process_all_parts"):
                    if not st.session_state.mineru_token:
                        st.error("Vui lòng nhập Mineru Token trước khi upload.")
                    else:
                        # Initialize empty result
                        st.session_state.result_text = ""
                        
                        for idx, part in enumerate(st.session_state.split_parts):
                            # Progress bar
                            progress_container = st.empty()
                            with progress_container.container():
                                st.write(f"Đang xử lý {part['name']} ({idx+1}/{len(st.session_state.split_parts)})...")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                status_text.text("Đang upload và phân tích...")
                                
                                # Process the part
                                result = upload_and_parse_pdf(
                                    part["data"],
                                    st.session_state.mineru_token,
                                    progress_callback=lambda p: progress_bar.progress(p/100)
                                )
                                
                                if result.startswith("Lỗi") or result.startswith("HTTP") or result.startswith("Mã lỗi"):
                                    status_text.text(f"Lỗi: {result}")
                                    time.sleep(2)  # Show error briefly
                                else:
                                    status_text.text("Đang hiệu đính với Gemini API...")
                                    
                                    # Call Gemini for correction if API key is provided
                                    if st.session_state.gemini_api_key:
                                        corrected_text = call_gemini_api(result, st.session_state.gemini_api_key)
                                        
                                        if corrected_text.startswith("Lỗi"):
                                            status_text.text(f"Cảnh báo: {corrected_text}")
                                            final_text = result
                                        else:
                                            final_text = corrected_text
                                    else:
                                        final_text = result
                                    
                                    # Append to existing result
                                    if st.session_state.result_text:
                                        new_text = f"\n\n--- Kết quả từ {part['name']} ---\n{final_text}"
                                        st.session_state.result_text += new_text
                                    else:
                                        st.session_state.result_text = f"--- Kết quả từ {part['name']} ---\n{final_text}"
                                    
                                    status_text.text("Phần này hoàn thành!")
                                    time.sleep(1)  # Show completion briefly
                            
                            # Clear progress container for next part
                            progress_container.empty()
                        
                        # Show final result
                        st.success("Đã xử lý tất cả các phần!")
                        st.markdown("### Kết quả")
                        st.markdown(st.session_state.result_text)
    
    # Results section and Word conversion
    if st.session_state.result_text:
        st.subheader("Kết quả")
        
        # Add tabs for viewing and downloading
        tab1, tab2 = st.tabs(["Xem kết quả", "Tải xuống"])
        
        with tab1:
            st.markdown(st.session_state.result_text)
        
        with tab2:
            st.write("Tải xuống kết quả")
            
            # Download as Markdown
            md_data = st.session_state.result_text.encode()
            st.download_button(
                label="Tải xuống dưới dạng Markdown",
                data=md_data,
                file_name=f"{os.path.splitext(st.session_state.pdf_name)[0]}.md",
                mime="text/markdown"
            )
            
            # Convert tables in markdown
            md_with_tables = convert_tables_to_md(st.session_state.result_text)
            
            # Download as plain text
            st.download_button(
                label="Tải xuống dưới dạng Text",
                data=md_with_tables.encode(),
                file_name=f"{os.path.splitext(st.session_state.pdf_name)[0]}.txt",
                mime="text/plain"
            )
            
            # Note about Word conversion
            st.info(
                "Lưu ý: Việc chuyển đổi sang Word yêu cầu Pandoc được cài đặt trên máy chủ. "
                "Trong ứng dụng web, chức năng này bị giới hạn. Bạn có thể tải xuống định dạng Markdown "
                "và sử dụng các công cụ chuyển đổi như Pandoc trên máy tính của bạn."
            )

# ------------------------------ Run the app ------------------------------
if __name__ == "__main__":
    main()
