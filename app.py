def main():
    create_directories() 
    
    with st.sidebar:
        st.image("Image/91APP_logo.png")
        selected = option_menu("",
        ["每頁商品數固定",'每頁商品數不固定','品名翻譯'],
        icons=['caret-right-fill','caret-right-fill','caret-right-fill'], menu_icon="robot", default_index=0,
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

            pdf_file = popover.file_uploader("上傳商品型錄 PDF", type=["pdf"], key="pdf_file_uploader")
            data_file = popover.file_uploader("上傳貨號檔 CSV/XLSX", type=["csv", "xlsx"], key="data_file_uploader")
            json_file = popover.file_uploader("上傳 Google Cloud 憑證", type=["json"], key="json_file_uploader")
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
        pdf_file = st.session_state.get('pdf_file', None)
        data_file = st.session_state.get('data_file', None)
        json_file = st.session_state.get('json_file', None)
        api_key = st.session_state.get('api_key', None)

    if json_file:
        temp_json_path = os.path.join("temp", json_file.name)
        with open(temp_json_path, "wb") as f:
            f.write(json_file.getbuffer())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path

    if selected == "每頁商品數固定":
        st.markdown('<div class="centered"><h2>商運  PDF截圖與AI文案生成工具</h2></div>', unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")
        st.text_input("指定截圖高度 (px)", placeholder="例如：255", value=st.session_state.height, key='height_input', on_change=update_height, help="如何找到截圖高度？\n\n1.截一張想要的圖片範圍 \n 2.上傳Photoshop，查看左側的圖片高度")
        st.text_area("給 ChatGPT 的 Prompt", height=250, value=st.session_state.user_input, key='user_input_input', on_change=update_user_input)
    elif selected == "每頁商品數不固定":
        st.markdown('<div class="centered"><h2>商運  PDF截圖與AI文案生成工具</h2></div>', unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")
        st.text_input("用來判斷截圖高度的符號或文字", placeholder="例如：$", value=st.session_state.symbol, key='symbol_input', on_change=update_symbol)
        col1, col2 = st.columns([1,1.9])
        col1.text_area("對應的截圖高度（px）", placeholder="數量：高度（用換行分隔）\n----------------------------------------\n2:350\n3:240", height=250, value=st.session_state.height_map_str, key='height_map_str_input', on_change=update_height_map_str, help="如何找到截圖高度？\n\n1.截一張想要的圖片範圍 \n 2.上傳Photoshop，查看左側的圖片高度")
        col2.text_area("給 ChatGPT 的 Prompt", height=250, value=st.session_state.user_input, key='user_input_input', on_change=update_user_input)
    elif selected == "品名翻譯":
        st.markdown('<div class="centered"><h2>品名翻譯工具</h2></div>', unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")
        def translate_product_name(product_name, knowledge_data):
            translations = {}
            lines = product_name.split('\n')
            for line in lines:
                if '：' in line:
                    type_name, eng_name = line.split('：', 1)
                    # Find corresponding Chinese name from the knowledge base
                    matching_row = knowledge_data[(knowledge_data['品名類型'] == type_name) & (knowledge_data['EXCEL資料'].str.lower() == eng_name.strip().lower())]
                    if not matching_row.empty:
                        translations[type_name] = matching_row.iloc[0]['中文名稱']
                    else:
                        translations[type_name] = eng_name.strip()
                else:
                    translations[line] = line
            return translations
        
        def trigger_download(data, filename):
            b64 = base64.b64encode(data).decode()
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
                        link.href = "data:text/csv;base64,{b64}";
                        link.download = "{filename}";
                        link.click();
                    }}
                </script>
                </head>
                </html>
            """, height=0)
            st.toast("執行完成 🥳 檔案已自動下載至您的電腦")

        col1,col2 = st.columns(2)
        with col1:
            knowledge_file = st.file_uploader("上傳翻譯對照表 CSV/XLSX", type=["xlsx", "csv"])
            with st.expander("品名對照表 範例格式"):
                example_knowledge_data = pd.read_csv("品名對照表範例格式.csv")
                ui.table(example_knowledge_data)
        
        with col2:
            test_file = st.file_uploader("上傳需要翻譯的檔案 CSV/XLSX", type=["xlsx", "csv"])
            with st.expander("翻譯品名 範例格式"):
                example_test_data = pd.read_csv("翻譯品名範例格式.csv")
                ui.table(example_test_data)
            
    
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
            
            # Extract the column names from the test data
            column_names = test_data.columns.to_list()
            
            for index, row in test_data.iterrows():
                product_translations = translate_product_name(row[column_names[1]], knowledge_data)  # assuming second column is the product name
                product_translations = {column_names[0]: row[column_names[0]], **product_translations}  # keep the first column at the first position
                translated_data.append(product_translations)
            
            translated_df = pd.DataFrame(translated_data)
            
            st.divider()
            st.write("翻譯結果")
            with st.container(height=300, border=None):
                ui.table(translated_df)
            
            # CSV download
            csv = translated_df.to_csv(index=False, encoding='utf-8-sig')
            csv_data = csv.encode('utf-8-sig')
            
            # 使用 trigger_download 函數自動下載CSV文件
            trigger_download(csv_data, '翻譯結果.csv')
            
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
        if selected == "每頁商品數固定" and not st.session_state.height:
            missing_fields.append("指定截圖高度")
        if selected == "每頁商品數不固定":
            if not st.session_state.symbol:
                missing_fields.append("用來判斷截圖高度的符號或文字")
            if not st.session_state.height_map:
                missing_fields.append("對應的截圖高度")
        return missing_fields
    
    # 檢查所有必需字段是否已填寫
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
                    if selected == "每頁商品數固定":
                        search_and_zip_case1(pdf_path, texts, int(st.session_state.height), output_dir, zipf, api_key, st.session_state.user_input)
                    elif selected == "每頁商品數不固定":
                        search_and_zip_case2(pdf_path, texts, st.session_state.symbol, st.session_state.height_map, output_dir, zipf)
    
                    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    data = []
                    total_files = len(image_files)
                    
                    st.write('\n')
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
            ui.metric_card(title="本次執行費用", content=f"${total_cost_twd:.2f} NTD", description="根據即時匯率", key="card3")
            
        with st.container(height=400,border=None):
            st.write("##### 成果預覽")
            ui.table(st.session_state.df_text)
        trigger_download(st.session_state.zip_buffer, "output.zip")
        st.session_state.download_triggered = True

        
if __name__ == "__main__":
    main()
