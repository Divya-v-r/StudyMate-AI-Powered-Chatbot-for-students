import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv()
# ---------------- Install required packages ----------------
def install_packages():
    packages = [
        'gradio',
        'google-generativeai',
        'pillow',
        'requests',
        'python-dotenv',
        'sentence-transformers',
        'faiss-cpu',
        'PyMuPDF'
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

install_packages()