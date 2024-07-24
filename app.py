import streamlit as st
import fitz  # PyMuPDF
import os
import shutil
import zipfile
import pandas as pd
from google.cloud import vision, language_v1
from PIL import Image, ImageEnhance
import io

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

def rename_img(old_p, new_name):
    new_p = os.path.join(os.path.dirname(old_p), new_name)
    os.rename(old_p, new_p)
    return new_p

def search_extract_img(file, text, out_dir, h, offset=0):
    res = search_pdf(file, text)
    if res:
        page_num, rect = res[0]
        img_p = extract_img(file, page_num, rect, out_dir, h=h, offset=offset)
        new_img_p = rename_img(img_p, f"{text}.png")
        return page_num, new_img_p
    return None, None

def search_and_zip_case1(file, texts, h, out_dir, zipf):
    total_files = len(texts)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("準備載入PDF與CSV文件")

    for i, text in enumerate(texts):
        page_num, img_p = search_extract_img(file, text, out_dir, h=h)
        if img_p:
            zipf.write(img_p, os.path.basename(img_p))
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    progress_bar.empty()
    progress_text.empty()

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
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        progress_text.text(f"正在擷取圖片: {text} ({i + 1}/{total_files})")
    progress_bar.empty()
    progress_text.empty()

def preprocess_image(img):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    return img

def extract_text_from_image(img_path):
    client = vision.ImageAnnotatorClient()
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts
    return []

def analyze_text_with_nlp(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document)
    entities = response.entities
    entity_texts = {entity.name: entity for entity in entities}
    return entity_texts

def process_and_match_text_blocks(text_blocks):
    if not text_blocks:
        return ""
    
    description = text_blocks[0].description
    blocks = text_blocks[1:]

    block_data = [(block.bounding_poly.vertices, block.description) for block in blocks]
    sorted_blocks = sorted(block_data, key=lambda x: (x[0][0].y, x[0][0].x))
    combined_text = " ".join([text for _, text in sorted_blocks])
    
    entities = analyze_text_with_nlp(combined_text)
    
    formatted_text = ""
    for entity_name, entity in entities.items():
        formatted_text += f"{entity_name}: {entity.name}\n"
    
    return formatted_text.strip()

if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None
if 'zip_file_ready' not in st.session_state:
    st.session_state.zip_file_ready = False
if 'df_text' not in st.session_state:
    st.session_state.df_text = pd.DataFrame()

def main():
    create_directories()

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
            clear_directory(output_dir)

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
                if option == "每頁商品數「固定」的情形":
                    search_and_zip_case1(pdf_path, texts, height, output_dir, zipf)
                else:
                    search_and_zip_case2(pdf_path, texts, symbol, height_map, output_dir, zipf)

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

                    text_blocks = extract_text_from_image(img_path)
                    formatted_text = process_and_match_text_blocks(text_blocks)
                    data.append({"貨號": os.path.splitext(image_file)[0], "商品資料": formatted_text})
                    
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
