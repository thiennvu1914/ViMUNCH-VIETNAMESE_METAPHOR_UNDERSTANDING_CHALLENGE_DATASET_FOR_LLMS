# ViMUNCH Demo - Phân tích Ẩn dụ Tiếng Việt

Demo Streamlit cho đồ án phân tích ẩn dụ tiếng Việt sử dụng model Vistral-7B-Chat.

## 🎯 Pipeline

| Task | Mô tả | Approach |
|------|-------|----------|
| **Task 1a** | Nhận diện ẩn dụ | Vistral Zero-shot / Fine-tuned |
| **Task 1b** | Trích xuất span | Vistral Zero-shot / Fine-tuned |
| **Task 2** | Phân loại ẩn dụ | Vistral Zero-shot / Fine-tuned |
| **Task 3** | Diễn giải | **Vistral Few-shot 5** |
| **Task 4** | Chấm điểm | Vistral Zero-shot / Fine-tuned |

## 📁 Cấu trúc

```
Demo_KLTN/
├── app/
│   ├── app.py                 # Main Streamlit app
│   └── utils/
│       ├── config.py          # Configuration
│       ├── model_loader.py    # Model loading
│       ├── inference.py       # Inference pipeline
│       ├── prompt_builder.py  # Prompt engineering
│       └── json_parser.py     # JSON extraction & validation
├── core/
│   ├── ViMUNCH/               # Dataset
│   │   ├── vimunch_train.json
│   │   ├── vimunch_dev.json
│   │   └── vimunch_test.json
│   ├── Result/
│   │   └── Vistral-7B-Chat_sft_multitask/  # Fine-tuned model (đặt vào đây)
│   └── ViMUNCH_Vistral-7B-Chat_Shot.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt & Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đảm bảo dataset ViMUNCH nằm tại:
- `core/ViMUNCH/vimunch_train.json`
- `core/ViMUNCH/vimunch_dev.json`
- `core/ViMUNCH/vimunch_test.json`

### 3. (Tùy chọn) Thêm Fine-tuned model

Đặt model đã fine-tune vào:
- `core/Result/Vistral-7B-Chat_sft_multitask/`

### 4. Chạy demo

```bash
cd app
streamlit run app.py
```

Hoặc từ thư mục gốc:

```bash
streamlit run app/app.py
```

## 💻 Yêu cầu hệ thống

- **GPU:** >= 16GB VRAM (cho 4-bit quantization)
- **RAM:** >= 16GB
- **Python:** >= 3.9

## 🔧 Cấu hình

Chỉnh sửa `app/utils/config.py` để thay đổi:
- Đường dẫn dataset
- Model name
- Đường dẫn fine-tuned model
- Các tham số inference

## 📝 Ghi chú

- Task 3 (Diễn giải) **luôn** sử dụng Few-shot 5 với base model
- Các task khác có thể chuyển đổi giữa Zero-shot và Fine-tuned qua checkbox trong sidebar
