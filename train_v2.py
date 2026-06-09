import os
import random
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập seed để tái lập kết quả
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# ==========================================
# 1. TẢI DỮ LIỆU & TĂNG CƯỜNG (NEGATIVE SAMPLING)
# ==========================================
print("📂 Đang tải dữ liệu từ checkpoint data_segmented.csv...")
csv_path = "data_segmented.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Không tìm thấy file '{csv_path}'. Vui lòng chạy tiền xử lý trước.")

df = pd.read_csv(csv_path)

def augment_negative_nei_samples(df, neg_ratio=0.30, random_state=42):
    """
    Tạo thêm các mẫu NEI nhân tạo bằng cách ghép cặp ngẫu nhiên Claim của hàng này 
    với Context của hàng khác không liên quan để mô hình học ranh giới NEI tốt hơn.
    """
    random.seed(random_state)
    original_size = len(df)
    num_negatives = int(original_size * neg_ratio)
    
    statements = df['Statement'].tolist()
    contexts = df['Context'].tolist()
    
    neg_samples = []
    for _ in range(num_negatives):
        idx_claim = random.randint(0, original_size - 1)
        idx_context = random.randint(0, original_size - 1)
        
        while idx_claim == idx_context:
            idx_context = random.randint(0, original_size - 1)
            
        neg_samples.append({
            'Unnamed: 0': 0,
            'index': 0,
            'Statement': statements[idx_claim],
            'Context': contexts[idx_context],
            'annotation_id': 0,
            'Topic': 'Augmented',
            'Author': 'Augmented',
            'Url': 'Augmented',
            'labels': 2,  # Gán nhãn NEI (2)
            'Evidence': "Không có thông tin liên quan trong bối cảnh.",
            'split': 'train',  # Chỉ thêm vào tập train
            'Statement_seg': df.loc[idx_claim, 'Statement_seg'] if 'Statement_seg' in df.columns else statements[idx_claim],
            'Evidence_seg': "Không_có thông_tin liên_quan trong bối_cảnh ."
        })
        
    df_neg = pd.DataFrame(neg_samples)
    df_augmented = pd.concat([df, df_neg], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_augmented

# Phân tách tập dữ liệu
train_df_raw = df[df['split'] == 'train'].reset_index(drop=True)
dev_df = df[df['split'] == 'dev'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"Kích thước tập huấn luyện gốc: {len(train_df_raw)}")
print("Đang tiến hành tăng cường dữ liệu NEI bằng Negative Sampling...")
train_df = augment_negative_nei_samples(train_df_raw, neg_ratio=0.30, random_state=42)
print(f"Kích thước tập huấn luyện sau tăng cường: {len(train_df)}")

# ==========================================
# 2. TÍNH TOÁN TRỌNG SỐ LỚP (CLASS WEIGHTS)
# ==========================================
# Tính toán class weights dựa trên nghịch đảo tần suất xuất hiện và thêm hệ số phạt
counts = train_df['labels'].value_counts()
total_samples = len(train_df)
weights = []
for i in range(3):
    count_i = counts.get(i, 1)
    weight_i = total_samples / (3.0 * count_i)
    weights.append(weight_i)

# Áp dụng hệ số phạt tùy chỉnh:
# Phạt nặng hơn khi đoán sai REFUTED (nhãn 1) và NEI (nhãn 2) để khắc phục sai lệch nghiêm trọng
weights[0] *= 1.0  # SUPPORTED
weights[1] *= 1.5  # REFUTED (phạt nặng hơn lỗi đối cực REFUTED -> SUPPORTED)
weights[2] *= 1.3  # NEI (phạt nặng hơn lỗi ép nhãn NEI -> REFUTED/SUPPORTED)

class_weights = torch.tensor(weights, dtype=torch.float)
print(f"⚖️ Trọng số phạt cho các lớp (Class Weights): {weights}")

# ==========================================
# 3. THIẾT LẬP DATASET & TOKENIZER
# ==========================================
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ViFactCheckDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        statement = str(self.df.loc[index, 'Statement_seg'])
        evidence = str(self.df.loc[index, 'Evidence_seg'])
        label = int(self.df.loc[index, 'labels'])

        encoding = self.tokenizer(
            statement,
            evidence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = ViFactCheckDataset(train_df, tokenizer, max_len=256)
dev_dataset = ViFactCheckDataset(dev_df, tokenizer, max_len=256)
test_dataset = ViFactCheckDataset(test_df, tokenizer, max_len=256)

# ==========================================
# 4. GHI ĐÈ HÀM LOSS TRONG HUGGING FACE TRAINER
# ==========================================
class CustomWeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Đẩy class weights lên cùng thiết bị chạy với logits
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 5. KHỞI TẠO MÔ HÌNH & HUẤN LUYỆN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Thiết bị sử dụng huấn luyện: {device}")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

trainer = CustomWeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer
)

print("🚀 Đang tiến hành huấn luyện mô hình PhoBERT v2...")
trainer.train()
print("✅ Huấn luyện thành công!")

# Lưu mô hình ở file mới an toàn tuyệt đối
os.makedirs("./models", exist_ok=True)
output_model_path = "./models/phobert_vifactcheck_v2.bin"
torch.save(model.state_dict(), output_model_path)
print(f"💾 Đã xuất file trọng số mới tại: {output_model_path}")

# ==========================================
# 6. ĐÁNH GIÁ TRÊN TẬP TEST & VẼ CONFUSION MATRIX
# ==========================================
print("\n🔍 Đang chạy suy luận trên tập kiểm thử (Test Set)...")
model.eval()

all_preds = []
all_labels = []

# Sử dụng DataLoader của PyTorch để suy luận nhanh
from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=16)

with torch.no_grad():
    for batch in test_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(b_labels.cpu().numpy())

LABEL_NAMES = ["SUPPORTED", "REFUTED", "NEI"]

print("\n" + "=" * 55)
print("KẾT QUẢ ĐÁNH GIÁ PHOBERT V2 TRÊN TEST SET")
print("=" * 55)
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))

# Tính F1-score để so sánh
f1_macro = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Macro V2: {f1_macro:.4f}")

# Vẽ Confusion Matrix mới
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.title('Confusion Matrix - PhoBERT v2 (Test Set)', fontweight='bold')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
plt.tight_layout()

cm_img_path = 'ket_qua_danh_gia_v2.png'
plt.savefig(cm_img_path, dpi=150)
print(f"📊 Đã lưu ma trận nhầm lẫn v2 tại: {cm_img_path}")
