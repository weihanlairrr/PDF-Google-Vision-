import streamlit as st
import fitz  # PyMuPDF
import os
import shutil
import zipfile
import pandas as pd
from google.cloud import vision
from PIL import Image, ImageEnhance
import io

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

# 定義從左到右、從上到下提取文字的函數
def extract_text_blocks(file):
    doc = fitz.open(file)
    texts = []
    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # 按照 y 坐標和 x 坐標排序
        for b in blocks:
            texts.append(b[4])
    return texts

# 定義搜尋多個文本並創建壓縮文件的函數
def search_and_zip(file, out_dir, zipf):
    texts = extract_text_blocks(file)
    total_files = len(texts)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("準備載入PDF與CSV文件")

    for i, text in enumerate(texts):
        page_num, img_p = search_extract_img(file, text, out_dir, h=60)
        if img_p:
            zipf.write(img_p, os.path.basename(img_p))
        # 更新進度條
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    progress_bar.empty()
    progress_text.empty()

# 定義圖像預處理函數
def preprocess_image(img):
    # 轉換為灰度圖像
    img = img.convert('L')
    # 增強對比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    return img

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

# 初始化 session state 變數
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None
if 'zip_file_ready' not in st.session_state:
    st.session_state.zip_file_ready = False
if 'df_text' not in st.session_state:
    st.session_state.df_text = pd.DataFrame()

def main():
    create_directories()  # 確保必要的目錄存在

    st.title("PDF截圖和文字提取工具")

    col1, col2 = st.columns(2)
    with col1:
        option = st.radio("選擇情況", ("每頁商品數「固定」的情形", "每頁商品數「不固定」的情形"), label_visibility="collapsed")

    with st.sidebar:
        pdf_file = st.file_uploader("上傳PDF文件", type=["pdf"])
        csv_file = st.file_uploader("上傳CSV文件", type=["csv"])
        json_file = st.file_uploader("上傳JSON憑證文件", type=["json"])

    if json_file:
        temp_json_path = os.path.join("temp", json_file.name)
        with open(temp_json_path, "wb") as f:
            f.write(json_file.getbuffer())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path

    if option == "每頁商品數「固定」的情形":
        height = st.text_area("指定截圖高度 (px)", placeholder="例如：255")
        if height:
            try:
                height = int(height)
            except ValueError:
                st.error("高度必須是數字。")
                return
    else:
        symbol = st.text_input("輸入用來判斷截圖高度的符號", placeholder="例如：$")
        height_map_str = st.text_area("輸入符號數量對應的截圖高度（格式：數量:高度，使用換行分隔）", placeholder="2:350\n3:240")
        height_map = {int(k): int(v) for k, v in (item.split(":") for item in height_map_str.split("\n") if item)}

    if pdf_file and csv_file and json_file:
        if st.button("開始執行"):
            temp_dir = "temp"
            output_dir = os.path.join(temp_dir, "output")
            clear_directory(output_dir)  # 清空 output 目錄

            pdf_path = os.path.join(temp_dir, pdf_file.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            csv_path = os.path.join(temp_dir, csv_file.name)
            with open(csv_path, "wb") as f:
                f.write(csv_file.getbuffer())

            df = pd.read_csv(csv_path, encoding='utf-8')
            texts = df.iloc[:, 0].tolist()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                search_and_zip(pdf_path, output_dir, zipf)

                image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                data = []
                total_files = len(image_files)

                progress_bar = st.progress(0)
                progress_text = st.empty()
                progress_text.text("準備載入截圖")

                for i, image_file in enumerate(image_files):
                    img_path = os.path.join(output_dir, image_file)
                    img = Image.open(img_path)
                    img = preprocess_image(img)

                    text = extract_text_from_image(img_path)
                    formatted_text = format_text(text)
                    data.append({"檔名": os.path.splitext(image_file)[0], "文字": formatted_text})
                    
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    progress_text.text(f"正在提取圖片文字: {image_file} ({i + 1}/{total_files})")

                progress_bar.empty()
                progress_text.empty()

                df_text = pd.DataFrame(data)
                csv_buffer = io.StringIO()
                df_text.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue().encode('utf-8')

                zipf.writestr("ocr_output.csv", csv_data)

            zip_buffer.seek(0)

            st.session_state.zip_buffer = zip_buffer.getvalue()
            st.session_state.zip_file_ready = True
            st.session_state.df_text = df_text

    if st.session_state.zip_file_ready and st.session_state.zip_buffer:
        st.dataframe(st.session_state.df_text)
        st.download_button(
            label="下載圖片和文字ZIP文件",
            data=st.session_state.zip_buffer,
            file_name="output.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
