import sys
import os
import transformers

# PHÒNG THỦ HỆ THỐNG & TERMINAL WINDOWS (MỞ ĐẦU FILE)
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass
transformers.logging.set_verbosity_error()

import re
import unicodedata
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

def local_css(file_name):
    """
    Nạp các tùy chỉnh CSS từ bên ngoài vào Streamlit để giữ giao diện sạch sẽ.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, file_name)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Không tìm thấy file CSS cấu hình tại {full_path}. Giao diện sẽ hiển thị mặc định.")

def clean_text(text) -> str:
    """
    Hàm làm sạch văn bản tiếng Việt:
    - Chuẩn hóa Unicode sang NFC.
    - Loại bỏ các thẻ HTML để làm sạch văn bản thô.
    - Chuẩn hóa đường dẫn URL và địa chỉ Email thành các token đặc biệt <URL> và <EMAIL> tương ứng.
    - Thay thế các ký tự xuống dòng (\n), tab (\t) bằng khoảng trắng và loại bỏ các khoảng trắng thừa.
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    
    text = str(text)
    
    # Chuẩn hóa Unicode tiếng Việt sang dạng NFC
    text = unicodedata.normalize("NFC", text)
    
    # Xóa các thẻ HTML nếu có
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Chuẩn hóa URL và Email thành các token đặc biệt
    text = re.sub(r"http\S+|www\S+", " <URL> ", text)
    text = re.sub(r"\S+@\S+", " <EMAIL> ", text)
    
    # Thay thế các ký tự xuống dòng, tab bằng khoảng trắng
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    
    # Loại bỏ khoảng trắng thừa ở đầu, cuối và giữa các từ
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def segment_uts(text: str) -> str:
    """
    Tách từ tiếng Việt bằng thư viện underthesea (Word Segmentation).
    Các từ ghép nhiều âm tiết sẽ được nối với nhau bằng dấu gạch dưới '_' (ví dụ: học_sinh).
    """
    if not text or len(text.strip()) == 0:
        return text
    
    # Thực hiện tách từ bằng underthesea và thay thế khoảng trắng bên trong từ ghép bằng '_'
    tokens = word_tokenize(text)
    segmented_tokens = [w.replace(" ", "_") for w in tokens]
    
    return " ".join(segmented_tokens)

def preprocess_text(text: str) -> str:
    """
    Hàm tiền xử lý chính cho một chuỗi văn bản:
    Nhận vào chuỗi văn bản thô, làm sạch ký tự thừa/đặc biệt và thực hiện tách từ tiếng Việt.
    """
    cleaned = clean_text(text)
    segmented = segment_uts(cleaned)
    return segmented

def preprocess_input(claim: str, context: str):
    """
    Hàm tổng hợp để kết hợp Claim (Nhận định) và Context (Bằng chứng/Ngữ cảnh) 
    theo yêu cầu xử lý đầu vào của mô hình PhoBERT.
    """
    processed_claim = preprocess_text(claim)
    processed_context = preprocess_text(context)
    return processed_claim, processed_context

# ==========================================
# KIẾN TRÚC TÌM KIẾM LAI HAI GIAI ĐOẠN
# (Hybrid Two-Stage Retrieval: BM25 + SBERT)
# ==========================================

@st.cache_resource
def load_sbert():
    """
    Nạp mô hình SBERT chuyên tiếng Việt một lần duy nhất và lưu trữ trong cache Streamlit.
    Mô hình 'keepitreal/vietnamese-sbert' được huấn luyện trên tập ngữ liệu tiếng Việt lớn,
    cho phép mã hóa câu thành vector ngữ nghĩa chất lượng cao.
    """
    return SentenceTransformer('keepitreal/vietnamese-sbert')

def advanced_evidence_retrieval(claim: str, context: str, top_k=4) -> str:
    """
    Kiến trúc Tìm kiếm Lai Hai Giai Đoạn (Hybrid Two-Stage Retrieval Pipeline):
    
    Giai đoạn 1 - Lọc thô bằng BM25 (Lexical Retrieval):
        Sử dụng thuật toán BM25Okapi để tính toán điểm trùng từ khóa giữa Claim và từng câu
        trong Context, lọc ra tối đa 10 câu ứng viên có điểm BM25 cao nhất.
    
    Giai đoạn 2 - Tinh lọc bằng SBERT (Semantic Re-ranking):
        Sử dụng mô hình Sentence-BERT tiếng Việt ('keepitreal/vietnamese-sbert') để mã hóa
        Claim và các câu ứng viên thành vector ngữ nghĩa, sau đó tính Cosine Similarity
        để xếp hạng lại và chọn ra top_k câu có độ tương đồng ngữ nghĩa cao nhất.
    
    Trả về:
        str: Chuỗi văn bản ghép từ top_k câu bằng chứng liên quan nhất.
    """
    if not claim.strip() or not context.strip():
        return context
    
    # Xé nhỏ Context thành danh sách câu bằng underthesea
    sentences = sent_tokenize(context)
    if len(sentences) <= top_k:
        return " ".join(sentences)
    
    # ---- GIAI ĐOẠN 1: Lọc thô bằng BM25 (Lấy top 10 câu trùng từ khóa) ----
    tokenized_corpus = [sent.lower().split(" ") for sent in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_claim = claim.lower().split(" ")
    bm25_top_n = min(10, len(sentences))
    bm25_scores = bm25.get_scores(tokenized_claim)
    top_indices_bm25 = np.argsort(bm25_scores)[::-1][:bm25_top_n]
    candidate_sentences = [sentences[i] for i in top_indices_bm25]
    
    # ---- GIAI ĐOẠN 2: Tinh lọc bằng SBERT (Re-rank ngữ nghĩa lấy top_k câu) ----
    sbert_model = load_sbert()
    claim_embedding = sbert_model.encode(claim, convert_to_tensor=True)
    corpus_embeddings = sbert_model.encode(candidate_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(claim_embedding, corpus_embeddings)[0]
    top_indices_sbert = np.argsort(cosine_scores.cpu().numpy())[::-1][:top_k]
    
    return " ".join([candidate_sentences[i] for i in top_indices_sbert])

# ==========================================
# NẠP CÁC MÔ HÌNH AI (CACHED)
# ==========================================

@st.cache_resource
def load_baseline_model():
    """
    Huấn luyện hoặc nạp mô hình Baseline (TF-IDF + Logistic Regression) để kiểm chứng đối chéo.
    Mô hình được huấn luyện động nhanh chóng (< 1 giây) từ file data_segmented.csv
    và lưu trữ qua Streamlit cache.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data_segmented.csv")
    if not os.path.exists(csv_path):
        return None, None
        
    try:
        df = pd.read_csv(csv_path)
        # Lấy dữ liệu thuộc tập huấn luyện để fit model
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        
        # Tạo text ghép cặp cho tập train
        train_df['combined_text'] = train_df['Statement_seg'].astype(str) + " " + train_df['Evidence_seg'].astype(str)
        
        # Cấu hình Vectorizer và Logistic Regression
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
        X_train = vectorizer.fit_transform(train_df['combined_text'])
        y_train = train_df['labels'].values
        
        model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        return model, vectorizer
    except Exception as e:
        # Ghi log nội bộ, không crash hệ thống
        sys.stderr.write(f"Warning: Không thể tự động khởi tạo mô hình Baseline: {e}\n")
        return None, None

@st.cache_resource
def load_model_and_tokenizer():
    """
    Nạp Tokenizer và Mô hình PhoBERT v2 đã huấn luyện nâng cấp từ thư mục ./models/
    Sử dụng @st.cache_resource để lưu trữ tài nguyên nặng (RAM) chỉ nạp 1 lần.
    """
    model_name = "vinai/phobert-base"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights_path = os.path.join(current_dir, "models", "phobert_vifactcheck_v2.bin")
    
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số mô hình PhoBERT v2 tại '{model_weights_path}'."
        )
        
    try:
        # Tải tokenizer tương ứng của PhoBERT
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Khởi tạo kiến trúc phân loại Sequence với 3 nhãn
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        
        # Xác định thiết bị chạy (GPU nếu có, ngược lại chạy CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Nạp state_dict (trọng số của mô hình v2 đã qua tối ưu hóa)
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Đưa mô hình lên thiết bị tương ứng và chuyển sang chế độ đánh giá
        model.to(device)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Lỗi nghiêm trọng khi khởi tạo mô hình PhoBERT v2: {str(e)}")

# ==========================================
# HÀM DỰ ĐOÁN (INFERENCE)
# ==========================================

def predict_factcheck(claim: str, context: str):
    """
    Dự đoán nhãn kiểm chứng cho nhận định (Claim) dựa trên ngữ cảnh/bằng chứng (Context)
    sử dụng mô hình PhoBERT v2 đã nạp.
    
    Trả về:
        predicted_label (str): Nhãn kết luận cuối cùng ("SUPPORTED", "REFUTED", hoặc "NEI").
        probs_dict (dict): Dictionary chứa xác suất (%) của cả 3 nhãn để phục vụ hiển thị/vẽ biểu đồ.
        retrieved_evidence (str): Đoạn ngữ cảnh rút gọn (evidence sạch) sau khi lọc context dài.
    """
    # 1. Rút trích văn bản dài bằng Kiến trúc Lai BM25 + SBERT (top_k=4 câu liên quan nhất)
    retrieved_evidence = advanced_evidence_retrieval(claim, context, top_k=4)
    
    # Bước A: Nạp model và tokenizer từ cache
    model, tokenizer = load_model_and_tokenizer()
    
    # Bước B: Tiền xử lý cặp chuỗi đầu vào tiếng Việt (sử dụng bối cảnh đã thu gọn)
    processed_claim, processed_context = preprocess_input(claim, retrieved_evidence)
    
    # Bước C: Mã hóa cặp chuỗi sang định dạng Tensor cho PyTorch
    inputs = tokenizer(
        processed_claim,
        processed_context,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    
    # Bước D: Đẩy các Tensor đầu vào lên cùng thiết bị chạy với mô hình (CPU hoặc GPU)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Bước E: Dự đoán không tính gradient để tiết kiệm tài nguyên
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Bước F: Lấy logits từ kết quả và chuyển đổi sang phân phối xác suất bằng hàm Softmax
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).flatten().cpu().numpy()
    
    # Bước G: Xác định nhãn kết luận có xác suất cao nhất
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    predicted_idx = probs.argmax()
    predicted_label = labels[predicted_idx]
    
    # Bước H: Tạo dictionary chứa xác suất chi tiết của từng nhãn
    probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    return predicted_label, probs_dict, retrieved_evidence

def predict_baseline(claim: str, retrieved_context: str):
    """
    Dự đoán nhãn kiểm chứng sử dụng mô hình Baseline TF-IDF + Logistic Regression.
    """
    lr_model, vectorizer = load_baseline_model()
    if lr_model is None or vectorizer is None:
        return "NEI", {"SUPPORTED": 0.0, "REFUTED": 0.0, "NEI": 1.0}
        
    # Tiền xử lý dữ liệu
    processed_claim = preprocess_text(claim)
    processed_context = preprocess_text(retrieved_context)
    
    # Ghép cặp văn bản dạng khoảng trắng giống khâu huấn luyện Baseline
    combined_text = f"{processed_claim} {processed_context}"
    
    # Biến đổi đặc trưng và dự đoán xác suất
    features = vectorizer.transform([combined_text])
    pred_idx = lr_model.predict(features)[0]
    probs = lr_model.predict_proba(features)[0]
    
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    predicted_label = labels[pred_idx]
    probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    return predicted_label, probs_dict

# ==========================================
# GIAO DIỆN CHÍNH (STREAMLIT UI)
# ==========================================

def main():
    st.set_page_config(
        page_title="ViFactCheck - Xác minh sự thật tiếng Việt",
        page_icon="🔍",
        layout="wide"
    )
    
    # Nạp CSS tùy chỉnh từ assets/style.css
    local_css("assets/style.css")
    
    st.markdown("<h1>🔍 ViFactCheck - Xác minh sự thật tiếng Việt</h1>", unsafe_allow_html=True)
    
    # Định nghĩa danh sách các kịch bản mẫu thử nghiệm đa lĩnh vực
    samples = {
        "Kinh tế: Tăng trưởng GDP năm 2023 (REFUTED)": {
            "claim": "Tăng trưởng GDP cả năm 2023 của Việt Nam đạt hơn 10%.",
            "context": "Báo cáo chính thức từ Tổng cục Thống kê công bố tăng trưởng GDP năm 2023 của Việt Nam ước tính đạt 5.05%. Mặc dù thấp hơn mục tiêu đề ra là 6.5%, đây vẫn là mức tăng trưởng tích cực so với nhiều quốc gia khác trong khu vực."
        },
        "Y tế: Gia tăng dịch tễ cúm A (SUPPORTED)": {
            "claim": "Sở Y tế ghi nhận số ca mắc cúm A gia tăng đột biến trong tháng qua.",
            "context": "Báo cáo dịch tễ mới nhất của Sở Y tế cho thấy số ca mắc cúm A tại địa bàn thành phố đang gia tăng đột biến trong tháng qua, với hơn 500 trường hợp nhập viện điều trị, tăng gấp đôi so với tháng trước đó."
        },
        "Xã hội: Mốc lịch sử dân số Việt Nam (SUPPORTED)": {
            "claim": "Việt Nam có 100 triệu dân vào năm 2023.",
            "context": "Dân số Việt Nam chính thức cán mốc 100 triệu người vào trung tuần năm 2023. Sự kiện này đánh dấu mốc lịch sử quan trọng, đưa Việt Nam trở thành quốc gia đông dân thứ 15 trên thế giới."
        },
        "Xã hội: Biến động giá vé máy bay (NEI)": {
            "claim": "Giá vé máy bay các chặng nội địa dịp lễ này sẽ giảm mạnh do có nhiều hãng bay mới tham gia thị trường.",
            "context": "Nhu cầu di chuyển bằng đường hàng không dịp nghỉ lễ tới đây dự kiến tăng cao. Các hãng hàng không đang khẩn trương chuẩn bị tăng chuyến bay để phục vụ hành khách."
        }
    }
    
    # Khởi tạo các giá trị session state cho Claim và Context nếu chưa có
    if "claim_val" not in st.session_state:
        st.session_state.claim_val = ""
    if "context_val" not in st.session_state:
        st.session_state.context_val = ""
        
    # Hàm callback cập nhật dữ liệu khi người dùng chọn mẫu test nhanh
    def update_inputs_from_sample():
        choice = st.session_state.selected_sample
        if choice != "Tùy chọn tự nhập...":
            st.session_state.claim_val = samples[choice]["claim"]
            st.session_state.context_val = samples[choice]["context"]
 
    # Kiểm tra sự tồn tại của tệp trọng số để hiển thị cảnh báo cấu hình ngay trên thanh sidebar
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights_path = os.path.join(current_dir, "models", "phobert_vifactcheck_v2.bin")
    if not os.path.exists(model_weights_path):
        st.sidebar.error(f"❌ CẢNH BÁO: Không tìm thấy tệp trọng số `{model_weights_path}`!")
        st.sidebar.info("Vui lòng đảm bảo bạn đã đặt tệp mô hình thích hợp vào đúng vị trí hoặc đã chạy kịch bản huấn luyện để sinh ra mô hình v2 trước khi chạy ứng dụng.")
    else:
        st.sidebar.success("🤖 Hệ thống PhoBERT v2: SẴN SÀNG")
        
    csv_path = os.path.join(current_dir, "data_segmented.csv")
    if not os.path.exists(csv_path):
        st.sidebar.warning(f"⚠️ Không tìm thấy `{csv_path}`. Mô hình Baseline sẽ bị vô hiệu hóa.")
    else:
        st.sidebar.success("📊 Mô hình Baseline TF-IDF: SẴN SÀNG")

    # Chia trang thành 2 cột với tỷ lệ 1.2 : 1
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<h3>📝 Dữ liệu kiểm tra</h3>", unsafe_allow_html=True)
        
        # Hộp chọn mẫu dữ liệu thử nghiệm nhanh
        st.selectbox(
            "🎯 Chọn nhanh mẫu dữ liệu thử nghiệm:",
            options=["Tùy chọn tự nhập..."] + list(samples.keys()),
            key="selected_sample",
            on_change=update_inputs_from_sample
        )
        
        # Ô nhập liệu tự động đồng bộ qua Session State
        claim = st.text_input(
            "Tuyên bố cần kiểm chứng (Claim):", 
            key="claim_val",
            placeholder="Nhập lời tuyên bố cần xác minh..."
        )
        context = st.text_area(
            "Ngữ cảnh / Bối cảnh chứa bằng chứng (Context):", 
            key="context_val",
            placeholder="Nhập đoạn văn bản ngữ cảnh chứa bằng chứng đối so sánh...",
            height=200
        )
        
        verify_btn = st.button("Tiến hành xác minh")
        
    with col2:
        st.markdown("<h3>📊 Kết quả phân tích ngữ nghĩa</h3>", unsafe_allow_html=True)
        if verify_btn:
            if not claim.strip() or not context.strip():
                st.warning("⚠️ Vui lòng nhập đầy đủ cả Lời tuyên bố và Ngữ cảnh bối cảnh!")
            else:
                with st.spinner("Đang thực hiện phân tích và đối sánh ngữ nghĩa..."):
                    try:
                        # 1. Nạp mô hình Baseline để lấy Vectorizer cho màng lọc TF-IDF
                        lr_model, vectorizer = load_baseline_model()
                        
                        # Cấu hình kiểm tra màng lọc ngữ nghĩa
                        skip_phobert = False
                        overlap_score = 1.0
                        
                        if vectorizer is not None:
                            # Tiền xử lý dữ liệu trước khi trích xuất vector đặc trưng
                            processed_claim = preprocess_text(claim)
                            processed_context = preprocess_text(context)
                            
                            # Biến đổi sang không gian Vector TF-IDF
                            claim_vec = vectorizer.transform([processed_claim])
                            context_vec = vectorizer.transform([processed_context])
                            
                            # Tính độ tương đồng ngữ nghĩa Cosine Similarity
                            from sklearn.metrics.pairwise import cosine_similarity
                            overlap_score = float(cosine_similarity(claim_vec, context_vec)[0][0])
                            
                            if overlap_score < 0.15:
                                # Kích hoạt (Short-circuit)
                                st.warning("🛡️ HỆ THỐNG ĐÃ KÍCH HOẠT KIỂM TRA TƯƠNG QUAN NGỮ NGHĨA")
                                st.info(f"Độ tương đồng ngữ nghĩa Cosine TF-IDF: {overlap_score:.4f} (Dưới ngưỡng an toàn 0.15)")
                                skip_phobert = True
                                label = "NEI"
                                base_label = "NEI"
                                probs = {"SUPPORTED": 0.0, "REFUTED": 0.0, "NEI": 1.0}
                                retrieved_evidence = "Không tìm thấy bằng chứng liên quan trong bối cảnh cung cấp."
                        
                        if not skip_phobert:
                            # 2. Trích xuất bằng chứng bằng Kiến trúc Lai BM25 + SBERT (top_k=4)
                            retrieved_evidence = advanced_evidence_retrieval(claim, context, top_k=4)
                            
                            # 3. Dự đoán bằng PhoBERT v2 (sử dụng bằng chứng đã trích xuất)
                            model, tokenizer = load_model_and_tokenizer()
                            processed_claim, processed_context = preprocess_input(claim, retrieved_evidence)
                            inputs = tokenizer(
                                processed_claim,
                                processed_context,
                                return_tensors="pt",
                                truncation=True,
                                max_length=256
                            )
                            device = next(model.parameters()).device
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = model(**inputs)
                            logits = outputs.logits
                            raw_probs = torch.softmax(logits, dim=1).flatten().cpu().numpy()
                            labels_list = ["SUPPORTED", "REFUTED", "NEI"]
                            predicted_idx = raw_probs.argmax()
                            label = labels_list[predicted_idx]
                            probs = {labels_list[i]: float(raw_probs[i]) for i in range(len(labels_list))}
                            
                            # 4. Dự đoán bằng TF-IDF Baseline (trên bối cảnh đã rút trích để nhất quán)
                            base_label, base_probs = predict_baseline(claim, retrieved_evidence)
                        
                        # 5. Hệ Thống Trọng Tài AI (Consensus Meter)
                        st.markdown("<h4>⚖️ Hệ Thống Trọng Tài AI (Consensus Meter)</h4>", unsafe_allow_html=True)
                        
                        # Lấy độ tự tin của nhãn do PhoBERT v2 chọn bằng cách ép kiểu
                        phobert_conf = float(probs[label])
                        
                        # Khởi tạo cờ kiểm soát xung đột chí mạng
                        disagreement_risk = False
                        
                        if label == base_label:
                            # Trường hợp 1: Đồng thuận
                            st.success("🤝 **ĐỒNG THUẬN CAO** - Cả hai kiến trúc Deep Learning (PhoBERT v2) và Machine Learning (TF-IDF Baseline) đều xác thực kết quả này.")
                        elif phobert_conf >= 0.65:
                            # Trường hợp 2: Xung đột nhẹ nhưng PhoBERT v2 tự tin cao
                            st.warning("⚠️ **PHÁT HIỆN XUNG ĐỘT LOGIC NHẸ giữa các mô hình**. Hệ thống ưu tiên kết luận từ PhoBERT v2 vì kiến trúc mạng Transformer (Attention) có khả năng nắm bắt ngữ nghĩa ngữ cảnh sâu rộng tốt hơn tần suất từ của Baseline (Độ chính xác F1-score tập Test đạt khoảng 82%), kết hợp với độ tin cậy cao của mô hình tại thời điểm suy luận.")
                        else:
                            # Trường hợp 3: Xung đột chí mạng và PhoBERT v2 tự tin thấp
                            disagreement_risk = True
                            st.error("🚨 HỆ THỐNG KHÔNG THỂ ĐỒNG THUẬN (High Risk Disagreement)")
                            st.warning("⚠️ **Hệ thống từ chối đưa ra kết luận cuối cùng** để đảm bảo tính khách quan do độ tự tin của mô hình PhoBERT v2 ở mức thấp (< 65%) và có sự lệch nhãn trực tiếp với mô hình Baseline. Khuyến nghị người dùng tự kiểm chứng lại bằng chứng ở hộp thông tin bên dưới.")
                            
                        # Hiển thị kết quả so sánh giữa 2 mô hình
                        col_ai1, col_ai2 = st.columns(2)
                        with col_ai1:
                            st.info(f"**PhoBERT v2**: `{label}` (Độ tin cậy: {phobert_conf*100:.2f}%)")
                        with col_ai2:
                            st.info(f"**TF-IDF Baseline**: `{base_label}`")
                        
                        # Sắp xếp xác suất của các nhãn PhoBERT v2 giảm dần
                        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                        
                        # Xác định các lớp CSS và nội dung tương ứng với nhãn kết quả
                        if disagreement_risk:
                            card_class = "card-nei"
                            icon = "🛡️"
                            card_title = "Từ chối đưa ra kết luận cuối cùng (High Risk)"
                            desc = "Kết quả phân tích không đạt được sự đồng thuận và độ tự tin của mô hình PhoBERT v2 quá thấp để tự quyết."
                        else:
                            card_title = f"Kết quả cuối cùng (PhoBERT v2): {label}"
                            if label == "SUPPORTED":
                                card_class = "card-supported"
                                icon = "✅"
                                desc = "Bằng chứng trong ngữ cảnh ỦNG HỘ lời tuyên bố này."
                            elif label == "REFUTED":
                                card_class = "card-refuted"
                                icon = "❌"
                                desc = "Bằng chứng trong ngữ cảnh BÁC BỎ lời tuyên bố này."
                            else:
                                card_class = "card-nei"
                                icon = "❓"
                                desc = "Ngữ cảnh không chứa đủ thông tin để kiểm chứng (Not Enough Information)."
                        
                        # Render Card kết quả tùy chỉnh bằng CSS
                        result_html = f"""
                        <div class="result-card {card_class}">
                            <div class="card-title">{icon} {card_title}</div>
                            <div class="card-desc">{desc}</div>
                        </div>
                        """
                        st.markdown(result_html, unsafe_allow_html=True)
                        
                        # Hiển thị Bằng chứng trích xuất (Evidence sạch)
                        st.markdown("<h4>📌 Bằng chứng đã trích xuất (Extracted Evidence)</h4>", unsafe_allow_html=True)
                        st.info(f"\"{retrieved_evidence}\"")
                        
                        # Render thanh tiến trình biểu diễn xác suất của từng nhãn (PhoBERT v2)
                        st.markdown("<h4>Độ tin cậy chi tiết (PhoBERT v2):</h4>", unsafe_allow_html=True)
                        for lbl, val in sorted_probs:
                            fill_class = f"fill-{lbl.lower()}"
                            prob_pct = val * 100
                            bar_html = f"""
                             <div class="progress-container">
                                <div class="progress-label">
                                    <span>{lbl}</span>
                                    <span>{prob_pct:.2f}%</span>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill {fill_class}" style="width: {prob_pct}%;"></div>
                                </div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Lỗi trong quá trình xử lý: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
        else:
            st.info("ℹ️ Vui lòng điền thông tin tuyên bố và ngữ cảnh ở cột bên trái, hoặc chọn mẫu dữ liệu thử nghiệm nhanh ở hộp chọn phía trên, sau đó nhấn nút 'Tiến hành xác minh' để xem kết quả phân tích ngữ nghĩa.")

if __name__ == "__main__":
    main()
