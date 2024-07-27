import streamlit as st
import fitz  # PyMuPDF
import os
import shutil
import zipfile
import pandas as pd
import io
import aiohttp
import asyncio
import concurrent.futures
import base64
import tiktoken
import streamlit_shadcn_ui as ui
import streamlit.components.v1 as components

from openai import OpenAI
from google.cloud import vision
from py_currency_converter import convert

with st.sidebar:
    st.markdown(
        """
        <style>
        .stTextInput, .stTextArea {
            box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.1);
            border-radius: 5px;
            border: none;
        }
        [data-testid='stFileUploader'] {
            width: 90%;
        }
        [data-testid='stFileUploader'] section button {
            background-color: black !important;
            color: white !important;
            border-radius: 5px;
        }
        [data-testid='stFileUploader'] section {
            background: black!important;
            color: black !important;
            padding: 0;
            float: left;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def create_directories():
    os.makedirs("static", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

# 定義在PDF中搜尋文本並返回頁碼和矩形區域的函數
def search_pdf(file, text):
    doc = fitz.open(file)
    res = []
    for i, page in enumerate(doc):
        insts = page.search_for(text)
        for inst in insts:
            res.append((i + 1, inst))
    return res

# 定義提取頁面特定區域作為圖片（全頁寬度），高解析度的函數
def extract_img(file, page_num, rect, out_dir, h, z=6.0, offset=0):
    doc = fitz.open(file)
    page = doc.load_page(page_num - 1)
    mat = fitz.Matrix(z, z)
    pw = page.rect.width
    clip = fitz.Rect(0, rect.y0 + offset, pw, rect.y0 + offset + h)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img_path = os.path.join(out_dir, f"page_{page_num}.png")
    pix.save(img_path)
    return img_path

# 定義重命名圖片文件的函數
def rename_img(old_p, new_name):
    new_p = os.path.join(os.path.dirname(old_p), new_name)
    os.rename(old_p, new_p)
    return new_p

# 定義搜尋文本並提取對應區域作為全頁寬度圖片的函數
def search_extract_img(file, text, out_dir, h, offset=0):
    res = search_pdf(file, text)
    if res:
        page_num, rect = res[0]
        img_p = extract_img(file, page_num, rect, out_dir, h=h, offset=offset)
        new_img_p = rename_img(img_p, f"{text}.png")
        return page_num, new_img_p
    return None, None

# 定義文本格式化函數
def format_text(text):
    lines = text.split('\n\n')
    formatted_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(formatted_lines)

# 使用 Google Vision API 提取文本
def extract_text_from_image(img_path):
    client = vision.ImageAnnotatorClient()
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    return ""

def trigger_download(zip_buffer, filename):
    b64 = base64.b64encode(zip_buffer).decode()
    components.html(f"""
        <html>
        <head>
        <script type="text/javascript">
            function downloadURI(uri, name) {{
                var link = document.createElement("a");
                link.href = uri;
                link.download = name;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}
            window.onload = function() {{
                var link = document.createElement("a");
                link.href = "data:application/zip;base64,{b64}";
                link.download = "{filename}";
                link.click();
            }}
        </script>
        </head>
        </html>
    """, height=0)

@st.cache_data
def get_upload_files():
    return {"pdf": None, "data": None, "json": None, "api_key": "", "height": "", "symbol": "", "height_map_str": "", "height_map": {}, "user_input": ""}

uploaded_files = get_upload_files()

def update_uploaded_files():
    if 'pdf_file_uploader' in st.session_state:
        uploaded_files["pdf"] = st.session_state.pdf_file_uploader
    if 'data_file_uploader' in st.session_state:
        uploaded_files["data"] = st.session_state.data_file_uploader
    if 'json_file_uploader' in st.session_state:
        uploaded_files["json"] = st.session_state.json_file_uploader
    if 'api_key' in st.session_state:
        uploaded_files["api_key"] = st.session_state.api_key
    if 'height_input' in st.session_state:
        uploaded_files["height"] = st.session_state.height_input
    if 'symbol_input' in st.session_state:
        uploaded_files["symbol"] = st.session_state.symbol_input
    if 'height_map_str_input' in st.session_state:
        uploaded_files["height_map_str"] = st.session_state.height_map_str_input
        height_map = {}
        for item in uploaded_files["height_map_str"].split("\n"):
            if ":" in item:
                k, v = item.split(":")
                height_map[int(k.strip())] = int(v.strip())
        uploaded_files["height_map"] = height_map
    if 'user_input_input' in st.session_state:
        uploaded_files["user_input"] = st.session_state.user_input_input

# 初始化 session state 變數
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None
if 'zip_file_ready' not in st.session_state:
    st.session_state.zip_file_ready = False
if 'df_text' not in st.session_state:
    st.session_state.df_text = pd.DataFrame()
if 'task_completed' not in st.session_state:
    st.session_state.task_completed = False
if 'download_triggered' not in st.session_state:
    st.session_state.download_triggered = False
if 'total_input_tokens' not in st.session_state:
    st.session_state.total_input_tokens = 0
if 'total_output_tokens' not in st.session_state:
    st.session_state.total_output_tokens = 0

async def fetch_gpt_response(session, api_key, text, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(text)}]
    }
    async with session.post(url, headers=headers, json=payload) as response:
        return await response.json()

async def process_texts(api_key, texts, prompt, batch_size=10):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks.extend([fetch_gpt_response(session, api_key, text, prompt) for text in batch])
            if tasks:
                results = await asyncio.gather(*tasks)
                for result, text in zip(results, batch):
                    organized_text = result['choices'][0]['message']['content']
                    formatted_text = format_text(organized_text)
                    yield {"貨號": text, "文案": formatted_text}

def search_and_zip_case1(file, texts, h, out_dir, zipf, api_key, prompt):
    total_files = len(texts)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("準備載入PDF與CSV文件")

    for i, text in enumerate(texts):
        page_num, img_p = search_extract_img(file, text, out_dir, h=h)
        if img_p:
            zipf.write(img_p, os.path.basename(img_p))
        # 更新進度條
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    progress_bar.empty()
    progress_text.empty()

# 定義搜尋多個文本並創建壓縮文件的函數，情況2
def search_and_zip_case2(file, texts, symbol, height_map, out_dir, zipf):
    total_files = len(texts)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("準備載入PDF與CSV文件")
    
    for i, text in enumerate(texts):
        res = search_pdf(file, text)
        if res:
            page_num, rect = res[0]
            doc = fitz.open(file)
            page = doc.load_page(page_num - 1)
            symbol_count = len(page.search_for(symbol))
            height = height_map.get(symbol_count, 240)
            img_p = extract_img(file, page_num, rect, out_dir, h=height, offset=-10)
            new_img_p = rename_img(img_p, f"{text}.png")
            zipf.write(new_img_p, os.path.basename(new_img_p))
        # 更新進度條
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    progress_bar.empty()
    progress_text.empty()

def update_height():
    if st.session_state['height'] != st.session_state['height_input']:
        st.session_state['height'] = st.session_state['height_input']

def update_user_input():
    if st.session_state['user_input'] != st.session_state['user_input_input']:
        st.session_state['user_input'] = st.session_state['user_input_input']

def update_symbol():
    if st.session_state['symbol'] != st.session_state['symbol_input']:
        st.session_state['symbol'] = st.session_state['symbol_input']

def update_height_map_str():
    if st.session_state['height_map_str'] != st.session_state['height_map_str_input']:
        st.session_state['height_map_str'] = st.session_state['height_map_str_input']
        height_map = {}
        for item in st.session_state['height_map_str'].split("\n"):
            if ":" in item:
                k, v = item.split(":")
                height_map[int(k.strip())] = int(v.strip())
        st.session_state['height_map'] = height_map

def main():
    create_directories() 
    option = ui.tabs(options=["每頁商品數「固定」的情形", "每頁商品數「不固定」的情形"], default_value="每頁商品數「固定」的情形")

    with st.sidebar:
        st.image("Image/91APP_logo.png")
        with st.expander("文件上傳"):
            pdf_file = st.file_uploader("上傳 PDF", type=["pdf"], key="pdf_file_uploader")
            data_file = st.file_uploader("上傳 CSV 或 XLSX", type=["csv", "xlsx"], key="data_file_uploader")
            json_file = st.file_uploader("上傳 Google Cloud 憑證", type=["json"], key="json_file_uploader")
        st.write("\n")
        api_key = st.text_input("輸入 OpenAI API Key", type="password")

    # 將已上傳的文件存入 session state
    if pdf_file:
        st.session_state.pdf_file = pdf_file
    if data_file:
        st.session_state.data_file = data_file
    if json_file:
        st.session_state.json_file = json_file
    if api_key:
        st.session_state.api_key = api_key

    # 從 session state 中獲取文件
    pdf_file = st.session_state.pdf_file
    data_file = st.session_state.data_file
    json_file = st.session_state.json_file
    api_key = st.session_state.api_key

    if json_file:
        temp_json_path = os.path.join("temp", json_file.name)
        with open(temp_json_path, "wb") as f:
            f.write(json_file.getbuffer())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path

    if option == "每頁商品數「固定」的情形":
        height = st.text_input("指定截圖高度 (px)", placeholder="例如：255", value=st.session_state.height, key='height_input', on_change=update_height, help="如何找到截圖高度？\n\n1.截一張想要的圖片範圍 \n 2.上傳Photoshop，查看左側的圖片高度")
        user_input = st.text_area("給 ChatGPT 的 Prompt", height=300, value=st.session_state.user_input, key='user_input_input', on_change=update_user_input)
    else:
        symbol = st.text_input("用來判斷截圖高度的符號或文字", placeholder="例如：$", value=st.session_state.symbol, key='symbol_input', on_change=update_symbol)
        col1, col2 = st.columns([1,1.9])
        height_map_str = col1.text_area("對應的截圖高度（px）", placeholder="數量：高度（用換行分隔）\n----------------------------------------\n2:350\n3:240", height=300, value=st.session_state.height_map_str, key='height_map_str_input', on_change=update_height_map_str, help="如何找到截圖高度？\n\n1.截一張想要的圖片範圍 \n 2.上傳Photoshop，查看左側的圖片高度")
        user_input = col2.text_area("給 ChatGPT 的 Prompt", height=300, value=st.session_state.user_input, key='user_input_input', on_change=update_user_input)
    
    def organize_text_with_gpt(text, api_key):
        client = OpenAI(api_key=api_key)
        prompt = f"'''{text} '''{st.session_state.user_input}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        # 使用 tiktoken 計算 tokens 數量
        encoding = tiktoken.encoding_for_model("gpt-4")
        input_tokens = len(encoding.encode(prompt))
        output_tokens = len(encoding.encode(response.choices[0].message.content))
        
        # 將 tokens 計數存入 session_state
        st.session_state.total_input_tokens += input_tokens
        st.session_state.total_output_tokens += output_tokens
        
        return response.choices[0].message.content
    
    def check_required_fields():
        missing_fields = []
        if not pdf_file:
            missing_fields.append("PDF")
        if not data_file:
            missing_fields.append("CSV 或 XLSX")
        if not json_file:
            missing_fields.append("Google Cloud 憑證")
        if not api_key:
            missing_fields.append("OpenAI API Key")
        if not st.session_state.user_input:
            missing_fields.append("給 ChatGPT 的 Prompt")
        if option == "每頁商品數「固定」的情形" and not st.session_state.height:
            missing_fields.append("指定截圖高度")
        if option == "每頁商品數「不固定」的情形":
            if not st.session_state.symbol:
                missing_fields.append("用來判斷截圖高度的符號或文字")
            if not st.session_state.height_map:
                missing_fields.append("對應的截圖高度")
        return missing_fields
    
    # 檢查所有必需字段是否已填寫
    missing_fields = check_required_fields()
    
    if ui.button("開始執行", key="run_btn"):
        if missing_fields:
            st.warning("請上傳或輸入以下必需的項目：{}".format("、".join(missing_fields)))
        else:
            # 重置輸入和輸出 tokens 計數
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0

            # 清除之前的顯示內容
            st.session_state.task_completed = False
            st.session_state.download_triggered = False
            st.session_state.zip_buffer = None
            st.session_state.zip_file_ready = False
            st.session_state.df_text = pd.DataFrame()

            temp_dir = "temp"
            output_dir = os.path.join(temp_dir, "output")
            clear_directory(output_dir)  

            pdf_path = os.path.join(temp_dir, uploaded_files["pdf"].name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_files["pdf"].getbuffer())

            data_path = os.path.join(temp_dir, uploaded_files["data"].name)
            with open(data_path, "wb") as f:
                f.write(uploaded_files["data"].getbuffer())

            try:
                if uploaded_files["data"].name.endswith('.csv'):
                    df = pd.read_csv(data_path, encoding='utf-8')
                else:
                    df = pd.read_excel(data_path, engine='openpyxl')
            except UnicodeDecodeError:
                if uploaded_files["data"].name.endswith('.csv'):
                    df = pd.read_csv(data_path, encoding='latin1')
                else:
                    df = pd.read_excel(data_path, engine='openpyxl')

            texts = df.iloc[:, 0].tolist()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                if option == "每頁商品數「固定」的情形":
                    search_and_zip_case1(pdf_path, texts, int(uploaded_files["height"]), output_dir, zipf, uploaded_files["api_key"], uploaded_files["user_input"])
                else:
                    search_and_zip_case2(pdf_path, texts, uploaded_files["symbol"], uploaded_files["height_map"], output_dir, zipf)

                image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                data = []
                total_files = len(image_files)

                progress_bar = st.progress(0)
                progress_text = st.empty()
                progress_text.text("準備載入截圖")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(extract_text_from_image, os.path.join(output_dir, image_file)): image_file for image_file in image_files}
                    for future in concurrent.futures.as_completed(futures):
                        image_file = futures[future]
                        try:
                            text = future.result()
                            organized_text = organize_text_with_gpt(text, uploaded_files["api_key"])
                            formatted_text = format_text(organized_text)
                            data.append({"貨號": os.path.splitext(image_file)[0], "圖片內容": text, "文案": formatted_text})
                        except Exception as exc:
                            print(f'{image_file} generated an exception: {exc}')

                        progress = len(data) / total_files
                        progress_bar.progress(progress)
                        progress_text.text(f"正在提取圖片文字與撰寫文案: {image_file} ({len(data)}/{total_files})")

                progress_bar.empty()
                progress_text.empty()

                df_text = pd.DataFrame(data)
                csv_buffer = io.StringIO()
                df_text.to_csv(csv_buffer, index=False, encoding='utf-8-sig')  
                csv_data = csv_buffer.getvalue().encode('utf-8-sig')

                zipf.writestr("ocr_output.csv", csv_data)

            zip_buffer.seek(0)

            st.session_state.zip_buffer = zip_buffer.getvalue()
            st.session_state.zip_file_ready = True
            st.session_state.df_text = df_text
            st.session_state.task_completed = True

    if st.session_state.task_completed and st.session_state.zip_file_ready and not st.session_state.download_triggered:
        def usd_to_twd(usd_amount):
            result = convert(base='USD', amount=usd_amount, to=['TWD'])
            return result['TWD']

        input_cost = st.session_state.total_input_tokens / 1_000_000 * 0.15
        output_cost = st.session_state.total_output_tokens / 1_000_000 * 0.60
        total_cost_usd = input_cost + output_cost
        total_cost_twd = usd_to_twd(total_cost_usd)
            
        st.toast("執行完成 🥳 檔案已自動下載至您的電腦")
        st.divider()
        col1,col2,col3 =st.columns(3)
        with col1:
            ui.metric_card(title="Input Tokens", content=f"{st.session_state.total_input_tokens} 個", description="US$0.15 / 每百萬個Tokens", key="card1")
        with col2:
            ui.metric_card(title="Output Tokens", content=f"{st.session_state.total_output_tokens} 個", description="US$0.60 / 每百萬個Tokens", key="card2")
        with col3:
            ui.metric_card(title="本次執行費用", content=f"${total_cost_twd:.2f} 台幣", description="根據即時匯率", key="card3")
            
        with st.container(height=400):
            st.write("##### 成果預覽")
            ui.table(st.session_state.df_text)
        trigger_download(st.session_state.zip_buffer, "output.zip")
        st.session_state.download_triggered = True

if __name__ == "__main__":
    main()
