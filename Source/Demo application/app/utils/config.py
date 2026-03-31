"""
Configuration settings for ViMUNCH Demo
Kế thừa từ notebook ViMUNCH_Vistral-7B-Chat_Shot.ipynb
"""
import os
import torch

# ====== ĐƯỜNG DẪN DỮ LIỆU ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "ViMUNCH")

TRAIN_PATH = os.path.join(DATA_DIR, "vimunch_train.json")
DEV_PATH = os.path.join(DATA_DIR, "vimunch_dev.json")
TEST_PATH = os.path.join(DATA_DIR, "vimunch_test.json")

# ====== HUGGINGFACE TOKEN ======
HF_TOKEN = "YOUR_HF_TOKEN"

# ====== MODEL ======
MODEL_NAME = "Viet-Mistral/Vistral-7B-Chat"
TRUST_REMOTE_CODE = False

# ====== FINE-TUNED MODEL PATH ======
# Đường dẫn đến LoRA adapter đã fine-tune
FT_MODEL_PATH = "/root/ViMUNCH/Vistral_FT"

# ====== OUTPUT ======
RESULT_ROOT = os.path.join(BASE_DIR, "core", "Result", "vistral_7b")

# ====== FIXED FEW-SHOT IDS (đảm bảo công bằng) ======
FEWSHOT_IDS = [4839, 758, 427, 8151, 7679]

# ====== LABEL SPACE ======
ALLOWED_TYPES = ["structural", "ontological", "orientational", "emotional", "cultural_folklore"]

# ====== INFERENCE CONFIG ======
torch.backends.cuda.matmul.allow_tf32 = True
DTYPE = torch.bfloat16
BATCH_SIZE = 1  # Demo chạy từng câu
MAX_NEW_TOKENS = 512
RETRY_MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
DO_SAMPLE = False

# ====== SCORE KEYS ======
SCORE_KEYS = [
    "accuracy", "clarity", "naturalness", 
    "meaning", "implication", "modality", "syntax", "context",
    "overall", "quality"
]

# ====== TYPE DESCRIPTIONS (cho UI) ======
TYPE_DESCRIPTIONS = {
    "structural": "Ẩn dụ cấu trúc (Lakoff & Johnson): Một lĩnh vực khái niệm được cấu trúc hóa theo một lĩnh vực khác, giúp hiểu khái niệm phức tạp thông qua khái niệm quen thuộc.",
    "orientational": "Ẩn dụ định hướng (Lakoff & Johnson): Diễn tả khái niệm trừu tượng qua phương hướng không gian (lên/xuống, vào/ra, trước/sau), định hình giá trị tích cực hay tiêu cực.",
    "ontological": "Ẩn dụ bản thể (Lakoff & Johnson): Xem khái niệm trừu tượng như vật thể hữu hình hoặc sinh thể, giúp dễ hình dung và tác động.",
    "cultural_folklore": "Ẩn dụ văn hóa dân gian: Bắt nguồn từ kinh nghiệm tập thể, truyền thống và văn hóa dân gian, phản ánh thế giới quan và bản sắc ngôn ngữ người Việt.",
    "emotional": "Ẩn dụ cảm xúc: Biểu đạt tinh tế trạng thái tình cảm, cảm xúc con người qua hình ảnh cụ thể, giàu sức gợi — nét đặc trưng giàu tính biểu cảm của tiếng Việt."
}
