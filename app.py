import re
import unicodedata
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize

def local_css(file_name):
    """
    Nạp các tùy chỉnh CSS từ bên ngoài vào Streamlit để giữ giao diện sạch sẽ.
    """
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def clean_text(text) -> str:
    """
    Hàm làm sạch văn bản tiếng Việt:
    - Chuẩn hóa Unicode sang NFC (NFC là chuẩn hóa chuẩn mà nhóm đang dùng).
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
    
    Trả về:
        Bộ đôi (tuple) gồm: (processed_claim, processed_context) đã được tiền xử lý và tách từ.
        Cặp chuỗi này sau đó có thể được truyền trực tiếp vào tokenizer của Hugging Face dạng:
        tokenizer(processed_claim, processed_context) để tự động bổ sung các token đặc biệt (<s>, </s>) một cách chính xác.
    """
    processed_claim = preprocess_text(claim)
    processed_context = preprocess_text(context)
    return processed_claim, processed_context

@st.cache_resource
def load_model_and_tokenizer():
    """
    Nạp Tokenizer và Mô hình PhoBERT đã huấn luyện từ thư mục ./models/
    Sử dụng @st.cache_resource để lưu trữ tài nguyên nặng (RAM) chỉ nạp 1 lần.
    """
    model_name = "vinai/phobert-base"
    model_weights_path = "./models/phobert_vifactcheck.bin"
    
    # Tải tokenizer tương ứng của PhoBERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Khởi tạo kiến trúc phân loại Sequence với 3 nhãn
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Xác định thiết bị chạy (GPU nếu có, ngược lại chạy CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Nạp state_dict (trọng số của mô hình) được lưu từ train_phobert.ipynb
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Đưa mô hình lên thiết bị tương ứng và chuyển sang chế độ đánh giá
    model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_factcheck(claim: str, context: str):
    """
    Dự đoán nhãn kiểm chứng cho nhận định (Claim) dựa trên ngữ cảnh/bằng chứng (Context)
    sử dụng mô hình PhoBERT đã nạp.
    
    Trả về:
        predicted_label (str): Nhãn kết luận cuối cùng ("SUPPORTED", "REFUTED", hoặc "NEI").
        probs_dict (dict): Dictionary chứa xác suất (%) của cả 3 nhãn để phục vụ hiển thị/vẽ biểu đồ.
    """
    # Bước A: Nạp model và tokenizer từ cache
    model, tokenizer = load_model_and_tokenizer()
    
    # Bước B: Tiền xử lý cặp chuỗi đầu vào tiếng Việt
    processed_claim, processed_context = preprocess_input(claim, context)
    
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
    
    return predicted_label, probs_dict

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
                        label, probs = predict_factcheck(claim, context)
                        
                        # Sắp xếp xác suất của các nhãn giảm dần
                        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                        
                        # Xác định các lớp CSS và nội dung tương ứng với nhãn dự đoán
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
                            <div class="card-title">{icon} Kết quả: {label}</div>
                            <div class="card-desc">{desc}</div>
                        </div>
                        """
                        st.markdown(result_html, unsafe_allow_html=True)
                        
                        # Render thanh tiến trình biểu diễn xác suất của từng nhãn
                        st.markdown("<h4>Độ tin cậy chi tiết của thuật toán:</h4>", unsafe_allow_html=True)
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
