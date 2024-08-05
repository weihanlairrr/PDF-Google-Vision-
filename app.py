import streamlit as st
import fitz  # PyMuPDF
import os
import shutil
import zipfile
import pandas as pd
import io
import re
import aiohttp
import asyncio
import concurrent.futures
import tiktoken
import streamlit_shadcn_ui as ui

from openai import OpenAI
from google.cloud import vision
from py_currency_converter import convert
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu

st.set_page_config(page_title=None, page_icon=None, layout="wide")

with st.sidebar:
    st.markdown(
        """
        <style>
        .stButton > button:hover {
            background: linear-gradient(135deg, #707070 0%, #707070 100%);      
        }
        .stDownloadButton button {
            background-color: #46474A !important;
            color: #f5f5f5 !important;
            border: none;
        }
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, #707070 0%, #707070 100%) !important;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            text-align: center;
        }
        [data-testid='stFileUploader'] section button {
            background: transparent !important;
            color: #46474A !important;
            border-radius: 5px;
            border: none;
            display: block;
            margin: 0 auto;
        }
        [data-testid='stFileUploader'] section {
            background: #ECECEC!important;
            color: black !重要;
            padding: 0;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        [data-testid=stSidebar] {
            background: #F9F9F9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# 初始化 session state 變數
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None
if 'zip_file_ready' not in st.session_state:
    st.session_state.zip_file_ready = False
if 'df_text' not in st.session_state:
    st.session_state.df_text = pd.DataFrame()
if 'pdf_file' not in st.session_state:
    st.session_state.pdf_file = None
if 'data_file' not in st.session_state:
    st.session_state.data_file = None
if 'json_file' not in st.session_state:
    st.session_state.json_file = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'height' not in st.session_state:
    st.session_state.height = ""
if 'width' not in st.session_state:
    st.session_state.width = ""
if 'symbol' not in st.session_state:
    st.session_state.symbol = ""
if 'height_map_str' not in st.session_state:
    st.session_state.height_map_str = ""
if 'height_map' not in st.session_state:
    st.session_state.height_map = {}
if 'width_map_str' not in st.session_state:
    st.session_state.width_map_str = ""
if 'width_map' not in st.session_state:
    st.session_state.width_map = {}
if 'user_input1' not in st.session_state:
    st.session_state.user_input1 = ""
if 'user_input2' not in st.session_state:
    st.session_state.user_input2 = "" 
if 'task_completed' not in st.session_state:
    st.session_state.task_completed = False
if 'total_input_tokens' not in st.session_state:
    st.session_state.total_input_tokens = 0
if 'total_output_tokens' not in st.session_state:
    st.session_state.total_output_tokens = 0
def create_directories():
    os.makedirs("static", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def search_pdf(file, text):
    doc = fitz.open(file)
    res = []
    for i, page in enumerate(doc):
        insts = page.search_for(text)
        for inst in insts:
            res.append((i + 1, inst))
    return res

def extract_img(file, page_num, rect, out_dir, h, w=None, z=6.0, offset=0):
    doc = fitz.open(file)
    page = doc.load_page(page_num - 1)
    mat = fitz.Matrix(z, z)
    pw = page.rect.width
    if w is None:
        clip = fitz.Rect(0, rect.y0 + offset, pw, rect.y0 + offset + h)
    else:
        clip = fitz.Rect(max(0, rect.x0 - 20), rect.y0 + offset, min(pw, rect.x0 - 20 + w), rect.y0 + offset + h)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img_path = os.path.join(out_dir, f"page_{page_num}.png")
    pix.save(img_path)
    return img_path

def rename_img(old_p, new_name):
    new_p = os.path.join(os.path.dirname(old_p), new_name)
    os.rename(old_p, new_p)
    return new_p

def search_extract_img(file, text, out_dir, h, w=None, offset=0):
    res = search_pdf(file, text)
    if res:
        page_num, rect = res[0]
        img_p = extract_img(file, page_num, rect, out_dir, h=h, w=w, offset=offset)
        new_img_p = rename_img(img_p, f"{text}.png")
        return page_num, new_img_p
    return None, None

def format_text(text):
    lines = text.split('\n\n')
    formatted_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(formatted_lines)

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

def search_and_zip_case1(file, texts, h, w, out_dir, zipf, api_key, prompt):
    total_files = len(texts)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("準備載入PDF與CSV文件")
    
    not_found = []

    for i, text in enumerate(texts):
        page_num, img_p = search_extract_img(file, text, out_dir, h=h, w=w)
        if img_p:
            zipf.write(img_p, os.path.basename(img_p))
        else:
            not_found.append(text)
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    progress_bar.empty()
    progress_text.empty()

    return not_found

def search_and_zip_case2(file, texts, symbol, height_map, width_map, out_dir, zipf):
    total_files = len(texts)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("準備載入PDF與CSV文件")

    doc = fitz.open(file)
    symbol_found = False
    
    not_found = []

    for page in doc:
        if page.search_for(symbol):
            symbol_found = True
            break

    if not symbol_found:
        st.warning(f"無法在PDF中找到 \"{symbol}\"")
        return

    for i, text in enumerate(texts):
        res = search_pdf(file, text)
        if res:
            page_num, rect = res[0]
            page = doc.load_page(page_num - 1)
            symbol_count = len(page.search_for(symbol))
            height = height_map.get(symbol_count, 240)
            width = width_map.get(symbol_count, None)
            img_p = extract_img(file, page_num, rect, out_dir, h=height, w=width, offset=-10)
            new_img_p = rename_img(img_p, f"{text}.png")
            zipf.write(new_img_p, os.path.basename(new_img_p))
        else:
            not_found.append(text)
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    
    progress_bar.empty()
    progress_text.empty()
    
    return not_found

def update_api_key():
    if st.session_state['api_key'] != st.session_state['api_key_input']:
        st.session_state['api_key'] = st.session_state['api_key_input']
        
def update_height():
    if st.session_state['height'] != st.session_state['height_input']:
        st.session_state['height'] = st.session_state['height_input']

def update_width():
    if st.session_state['width'] != st.session_state['width_input']:
        st.session_state['width'] = st.session_state['width_input']

def update_user_input1():
    if st.session_state['user_input1'] != st.session_state['user_input_input1']:
        st.session_state['user_input1'] = st.session_state['user_input_input1']

def update_user_input2():
    if st.session_state['user_input2'] != st.session_state['user_input_input2']:
        st.session_state['user_input2'] = st.session_state['user_input_input2']
        
def update_symbol():
    if st.session_state['symbol'] != st.session_state['symbol_input']:
        st.session_state['symbol'] = st.session_state['symbol_input']

def update_height_map_str():
    st.session_state.height_map_errors = []  
    if st.session_state['height_map_str'] != st.session_state['height_map_str_input']:
        st.session_state['height_map_str'] = st.session_state['height_map_str_input']
        height_map = {}
        for item in st.session_state['height_map_str'].split("\n"):
            if ":" in item:
                try:
                    k, v = item.split(":")
                    height_map[int(k.strip())] = int(v.strip())
                except ValueError:
                    st.session_state.height_map_errors.append(f"無效的高度對應輸入: {item}")
        st.session_state['height_map'] = height_map

def update_width_map_str():
    st.session_state.width_map_errors = [] 
    if st.session_state['width_map_str'] != st.session_state['width_map_str_input']:
        st.session_state['width_map_str'] = st.session_state['width_map_str_input']
        width_map = {}
        for item in st.session_state['width_map_str'].split("\n"):
            if ":" in item:
                try:
                    k, v = item.split(":")
                    width_map[int(k.strip())] = int(v.strip())
                except ValueError:
                    st.session_state.width_map_errors.append(f"無效的寬度對應輸入: {item}")
        st.session_state['width_map'] = width_map

def split_content(row):
    content = row['文案']
    themes = re.split(r'【([^】]+)】', content)
    result = []
    
    for i in range(1, len(themes), 2):
        theme = themes[i]
        titles = re.split(r'〖([^〗]+)〗', themes[i+1])
        for j in range(1, len(titles), 2):
            title = titles[j]
            texts = re.split(r'\d\.\s', titles[j+1])
            texts = [text.strip() for text in texts if text.strip()]
            row_result = [row['貨號'], row['圖片內容'], theme, title] + texts
            result.append(row_result)
    
    return result

def split_columns(df):
    split_data = []

    for idx, row in df.iterrows():
        split_data.extend(split_content(row))

    columns = ['貨號', '圖片內容', '主題', '標題'] + [f'文案{i}' for i in range(1, max(len(row) for row in split_data) - 3)]

    result_df = pd.DataFrame(split_data, columns=columns)
    
    result_df['圖片內容'] = result_df['圖片內容'].mask(result_df['圖片內容'].duplicated(), '')

    return result_df

def main():
    create_directories() 
    
    with st.sidebar:
        st.image("Image/91APP_logo.png")
        selected = option_menu("",
        ["PDF截圖與AI文案",'品名翻譯'],
        icons=['caret-right-fill','caret-right-fill'], menu_icon="robot", default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "#F9F9F9","border-radius": "0px"},
            "icon": {"padding": "0px 10px 0px 0px !important","color": "#FF8C00", "font-size": "17px"},
            "nav-link": {"font-size": "17px","color": "#46474A", "text-align": "left", "margin":"0px", "--hover-color": "#f0f0f0"},
            "nav-link-selected": { "border-radius": "0px","background": "#EAE9E9", "color": "#2b2b2b"},
        }
    )

        if selected != "品名翻譯":
            with stylable_container(
                    key="popover",
                    css_styles="""
                        button {
                            background: #46474A;
                            color: white;
                            border-radius: 8px;
                            border: none;
                            width: 100%;
                            transition: background-color 0.3s;
                        }
                    """,
                ):
                st.write('\n')
                popover = st.popover("文件上傳")

            pdf_file = popover.file_uploader("上傳商品型錄 PDF", type=["pdf"], key="pdf_file_uploader",help="記得先刪除封面、目錄和多餘頁面")
            data_file = popover.file_uploader("上傳貨號檔 CSV 或 XLSX", type=["csv", "xlsx"], key="data_file_uploader",help="貨號放A欄，且首列須為任意標題")
            json_file = popover.file_uploader("上傳 Google Cloud 憑證 JSON", type=["json"], key="json_file_uploader")
            st.write("\n")
            with stylable_container(
                key="text_input_styles",
                css_styles="""
                    label {
                        color: #46474a;
                    }
                    """
                ):
                api_key = st.text_input("輸入 OpenAI API Key", type="password",key="api_key_input",on_change=update_api_key,value=st.session_state.api_key)

            if pdf_file:
                st.session_state.pdf_file = pdf_file
            if data_file:
                st.session_state.data_file = data_file
            if json_file:
                st.session_state.json_file = json_file
            if api_key:
                st.session_state.api_key = api_key

        pdf_file = st.session_state.get('pdf_file', None)
        data_file = st.session_state.get('data_file', None)
        json_file = st.session_state.get('json_file', None)
        api_key = st.session_state.get('api_key', None)

    if json_file:
        temp_json_path = os.path.join("temp", json_file.name)
        with open(temp_json_path, "wb") as f:
            f.write(json_file.getbuffer())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path

    if selected == "PDF截圖與AI文案":
        options = ui.tabs(options=[ "每頁商品數固定","每頁商品數不固定"], default_value= "每頁商品數固定", key="tabs")
        if options == "每頁商品數固定":
            col1 ,col2 = st.columns(2)
            col1.text_input("指定截圖高度 (px)", placeholder="例如：255", value=st.session_state.height, key='height_input', on_change=update_height, help="如何找到截圖高度？\n\n1.截一張想要的圖片範圍 \n 2.上傳Photoshop，查看左側的圖片高度")
            col2.text_input("指定截圖寬度 (px)", placeholder="未填則預設為完整PDF頁寬", value=st.session_state.width, key='width_input', on_change=update_width,help="選填")
            st.text_area("給 ChatGPT 的 Prompt",height=286, value=st.session_state.user_input1, key='user_input_input1', on_change=update_user_input1)
        elif  options == "每頁商品數不固定":  
            col1, col2 = st.columns([1,1.7])
            col1.text_input("用來判斷截圖高度的符號或文字", placeholder="例如：$", value=st.session_state.symbol, key='symbol_input', on_change=update_symbol)
            col1.text_area("對應的截圖高度（px）", placeholder="數量：高度（用換行分隔）\n----------------------------------------\n2:350\n3:240", height=120, value=st.session_state.height_map_str, key='height_map_str_input', on_change=update_height_map_str, help="如何找到截圖高度？\n\n1.截一張想要的圖片範圍 \n 2.上傳Photoshop，查看左側的圖片高度")
            col1.text_area("對應的截圖寬度（px）", placeholder="未填則預設為完整PDF頁寬，寫法同截圖高度", height=120, value=st.session_state.width_map_str, key='width_map_str_input', on_change=update_width_map_str, help="選填")
            col2.text_area("給 ChatGPT 的 Prompt",height=370, value=st.session_state.user_input2, key='user_input_input2', on_change=update_user_input2)
    
    elif selected == "品名翻譯":
        def translate_product_name(product_name, knowledge_data):
            translations = {}
            lines = product_name.split('\n')
            for line in lines:
                if '：' in line:
                    type_name, eng_name = line.split('：', 1)
                    matching_row = knowledge_data[(knowledge_data.iloc[:, 0] == type_name) & (knowledge_data.iloc[:, 1].str.lower() == eng_name.strip().lower())]
                    if not matching_row.empty:
                        translations[type_name] = matching_row.iloc[0, 2]
                    else:
                        translations[type_name] = eng_name.strip()
                else:
                    translations[line] = line
            return translations
        
        def load_data(file):
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                return pd.read_excel(file, sheet_name=None)
    
        col1, col2 = st.columns(2)
        with col1:
            knowledge_file = st.file_uploader("上傳翻譯對照表 CSV/XLSX", type=["xlsx", "csv"])
            with st.expander("品名對照表 範例格式"):
                example_knowledge_data = pd.read_csv("品名對照表範例格式.csv")
                ui.table(example_knowledge_data)
                example_knowledge_csv = example_knowledge_data.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="下載範例檔案", data=example_knowledge_csv, file_name="品名對照表範例格式.csv", mime="text/csv")
        
        with col2:
            test_file = st.file_uploader("上傳需要翻譯的檔案 CSV/XLSX", type=["xlsx", "csv"])
            with st.expander("待翻譯品名 範例格式"):
                example_test_data = pd.read_csv("翻譯品名範例格式.csv")
                ui.table(example_test_data)
                example_test_csv = example_test_data.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="下載範例檔案", data=example_test_csv, file_name="翻譯品名範例格式.csv", mime="text/csv")
        
        if knowledge_file and test_file:
            knowledge_data = load_data(knowledge_file)
            if isinstance(knowledge_data, dict):
                knowledge_data = knowledge_data[list(knowledge_data.keys())[0]]
        
            test_data = load_data(test_file)
        
            if isinstance(test_data, dict):
                test_data = test_data[list(test_data.keys())[0]]
        
            if not isinstance(test_data, pd.DataFrame):
                st.error("無法讀取測試檔案，請檢查檔案格式是否正確。")
                return
            translated_data = [] 
            column_names = test_data.columns.to_list()
            for index, row in test_data.iterrows():
                product_translations = translate_product_name(row[column_names[1]], knowledge_data)  
                product_translations = {column_names[0]: row[column_names[0]], **product_translations} 
                translated_data.append(product_translations)
        
            translated_df = pd.DataFrame(translated_data)

            placeholder1 = st.empty()
            placeholder2 = st.empty()
            placeholder3 = st.empty()

            with placeholder1.container():
                st.divider()
                st.write("翻譯結果")
                
            with placeholder2.container(height=400, border=None):
                ui.table(translated_df)

            with placeholder3.container():
                csv = translated_df.to_csv(index=False, encoding='utf-8-sig')
                csv_data = csv.encode('utf-8-sig')
                下載csv = st.download_button(
                    label="下載 CSV 檔案",
                    data=csv_data,
                    file_name="翻譯結果.csv",
                    mime="text/csv"
                )
            if 下載csv:
                placeholder1.empty()
                placeholder2.empty()
                placeholder3.empty()
                
    def organize_text_with_gpt(text, api_key):
        client = OpenAI(api_key=api_key)
        if selected == "PDF截圖與AI文案" and options == "每頁商品數固定":
            prompt = f"'''{text} '''{st.session_state.user_input1}"
        elif selected == "PDF截圖與AI文案" and options == "每頁商品數不固定":
            prompt = f"'''{text} '''{st.session_state.user_input2}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        encoding = tiktoken.encoding_for_model("gpt-4")
        input_tokens = len(encoding.encode(prompt))
        output_tokens = len(encoding.encode(response.choices[0].message.content))
        
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
        if selected == "PDF截圖與AI文案" and options == "每頁商品數固定"  and not st.session_state.user_input1:
            missing_fields.append("給 ChatGPT 的 Prompt")
        if selected == "PDF截圖與AI文案" and options == "每頁商品數不固定"  and not st.session_state.user_input2:
            missing_fields.append("給 ChatGPT 的 Prompt")
        if selected == "PDF截圖與AI文案" and options == "每頁商品數固定" and not st.session_state.height:
            missing_fields.append("指定截圖高度")
        if selected == "PDF截圖與AI文案" and options == "每頁商品數不固定":
            if not st.session_state.symbol:
                missing_fields.append("用來判斷截圖高度的符號或文字")
            if not st.session_state.height_map:
                missing_fields.append("對應的截圖高度")
        return missing_fields
    
    missing_fields = check_required_fields()
    if selected != "品名翻譯":
        with stylable_container(
                key="run_btn",
                css_styles="""
                    button {
                        background-color: #46474A;
                        color: white;
                        border-radius: 8px;
                        border: none;
                        width: 25%;
                    }
                    button:hover {
                        background: #6B6C70;
                    }
                    """,
                ):
            start_running = st.button("開始執行", key="run_btn")

        if start_running:
            if missing_fields:
                st.warning("請上傳或輸入以下必需的項目：{}".format("、".join(missing_fields)))
            else:
                st.write('\n')
                st.session_state.total_input_tokens = 0
                st.session_state.total_output_tokens = 0
    
                st.session_state.task_completed = False
                st.session_state.zip_buffer = None
                st.session_state.zip_file_ready = False
                st.session_state.df_text = pd.DataFrame()
    
                temp_dir = "temp"
                output_dir = os.path.join(temp_dir, "output")
                clear_directory(output_dir)  
    
                pdf_path = os.path.join(temp_dir, pdf_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
    
                data_path = os.path.join(temp_dir, data_file.name)
                with open(data_path, "wb") as f:
                    f.write(data_file.getbuffer())
    
                try:
                    if data_file.name.endswith('.csv'):
                        df = pd.read_csv(data_path, encoding='utf-8')
                    else:
                        df = pd.read_excel(data_path, engine='openpyxl')
                except UnicodeDecodeError:
                    if data_file.name.endswith('.csv'):
                        df = pd.read_csv(data_path, encoding='latin1')
                    else:
                        df = pd.read_excel(data_path, engine='openpyxl')
    
                texts = df.iloc[:, 0].tolist()
    
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                    not_found = []
                    if options == "每頁商品數固定":
                        not_found = search_and_zip_case1(pdf_path, texts, int(st.session_state.height), int(st.session_state.width) if st.session_state.width else None, output_dir, zipf, api_key, st.session_state.user_input1)
                    elif options == "每頁商品數不固定":
                        doc = fitz.open(pdf_path)
                        symbol_found = False

                        for page in doc:
                            if page.search_for(st.session_state.symbol):
                                symbol_found = True
                                break

                        if not symbol_found:
                            st.warning(f"無法在PDF中找到 \"{st.session_state.symbol}\"")
                            return

                        not_found = search_and_zip_case2(pdf_path, texts, st.session_state.symbol, st.session_state.height_map, st.session_state.width_map, output_dir, zipf)
                    
                    if not_found:
                        df_not_found = pd.DataFrame(not_found, columns=["無法搜尋到的貨號"])
                        csv_buffer_not_found = io.StringIO()
                        df_not_found.to_csv(csv_buffer_not_found, index=False, encoding='utf-8-sig')
                        csv_data_not_found = csv_buffer_not_found.getvalue().encode('utf-8-sig')
                        zipf.writestr("無法搜尋到的貨號.csv", csv_data_not_found)
                        st.warning("有特定貨號不存在於PDF中，請參考ZIP檔中的錯誤說明檔。")
    
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
                                organized_text = organize_text_with_gpt(text, api_key)
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
    
                    zipf.writestr("文字提取結果與文案.csv", csv_data)
    
                zip_buffer.seek(0)
    
                st.session_state.zip_buffer = zip_buffer.getvalue()
                st.session_state.zip_file_ready = True
                st.session_state.df_text = df_text
                st.session_state.task_completed = True

                if st.session_state.task_completed and st.session_state.zip_file_ready:
                    st.session_state.df_text = split_columns(st.session_state.df_text)
                    missing_items = [item for item in texts if item not in st.session_state.df_text['貨號'].unique()]
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                        for img_file in os.listdir(output_dir):
                            img_path = os.path.join(output_dir, img_file)
                            zipf.write(img_path, os.path.basename(img_path))
                            
                        csv_buffer = io.StringIO()
                        st.session_state.df_text.to_csv(csv_buffer, index=False, encoding='utf-8-sig')  
                        csv_data = csv_buffer.getvalue().encode('utf-8-sig')
                        zipf.writestr("文字提取結果與文案.csv", csv_data)
            
                        if missing_items:
                            missing_df = pd.DataFrame(missing_items, columns=['無法搜尋到的貨號'])
                            missing_csv_buffer = io.StringIO()
                            missing_df.to_csv(missing_csv_buffer, index=False, encoding='utf-8-sig')
                            missing_csv_data = missing_csv_buffer.getvalue().encode('utf-8-sig')
                            zipf.writestr("無法搜尋到的貨號.csv", missing_csv_data)
            
                    st.session_state.zip_buffer = zip_buffer.getvalue()
            
                    def usd_to_twd(usd_amount):
                        result = convert(base='USD', amount=usd_amount, to=['TWD'])
                        return result['TWD']
            
                    input_cost = st.session_state.total_input_tokens / 1_000_000 * 0.15
                    output_cost = st.session_state.total_output_tokens / 1_000_000 * 0.60
                    total_cost_usd = input_cost + output_cost
                    total_cost_twd = usd_to_twd(total_cost_usd)
                
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ui.metric_card(title="Input Tokens", content=f"{st.session_state.total_input_tokens} 個", description="US$0.15 / 每百萬個Tokens", key="card1")
                    with col2:
                        ui.metric_card(title="Output Tokens", content=f"{st.session_state.total_output_tokens} 個", description="US$0.60 / 每百萬個Tokens", key="card2")
                    with col3:
                        ui.metric_card(title="本次執行費用", content=f"${total_cost_twd:.2f} NTD", description="根據即時匯率", key="card3")
                
                    with st.container(height=400, border=None):
                        st.write("##### 成果預覽")
                        ui.table(st.session_state.df_text)
                    
                    st.download_button(
                        label="下載 ZIP 檔案",
                        data=st.session_state.zip_buffer,
                        file_name="output.zip",
                        mime="application/zip"
                    )
                    
if __name__ == "__main__":
    main()
