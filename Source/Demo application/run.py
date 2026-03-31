#!/usr/bin/env python3
"""
ViMUNCH Demo Launcher
Khởi chạy Streamlit server với thông báo rõ ràng
Model sẽ được load và cache bởi Streamlit (@st.cache_resource)
"""
import os
import sys
import subprocess

def log(msg):
    """Log ra stderr để hiển thị ngay lập tức"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def main():
    log("=" * 60)
    log("🚀 ViMUNCH Demo Launcher")
    log("=" * 60)
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = os.path.join(BASE_DIR, "app")
    app_path = os.path.join(APP_DIR, "app.py")
    
    # Check fine-tuned model
    sys.path.insert(0, APP_DIR)
    from utils.config import FT_MODEL_PATH
    
    ft_exists = os.path.exists(FT_MODEL_PATH)
    log(f"📂 Fine-tuned model path: {FT_MODEL_PATH}")
    log(f"📂 Fine-tuned model exists: {ft_exists}")
    log("")
    log("💡 Model sẽ được load khi truy cập web lần đầu tiên")
    log("   (Streamlit sử dụng @st.cache_resource để cache model)")
    log("")
    log("=" * 60)
    log("🌐 Đang khởi động Streamlit server...")
    log("=" * 60)
    log("")
    
    # Chạy streamlit
    os.chdir(BASE_DIR)
    cmd = [sys.executable, "-m", "streamlit", "run", app_path,
           "--server.port", "8501", "--server.address", "0.0.0.0"]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
