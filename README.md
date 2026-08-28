# ViMUNCH - A Benchmark Dataset for Evaluating Vietnamese Metaphor Understanding in LLMs

**Authors:** Hồ Nguyễn Thiên Vũ, Nguyễn Phi Long  
**Supervisor:** Dr. Nguyễn Văn Kiệt

---

## 📖 Introduction
**ViMUNCH** (Vietnamese Metaphor Understanding Challenge) is the first challenge dataset specifically designed to evaluate the capability of Large Language Models (LLMs) in understanding and interpreting Vietnamese metaphors. Unlike English datasets, ViMUNCH focuses on the linguistic and cultural characteristics of Vietnam, providing a standardized benchmark to verify models' cross-domain mapping abilities rather than just relying on surface lexical similarities.

## 🎥 Video Demo
Watch the practical demonstration of our analysis pipeline and project application here:  
[ViMUNCH Video Demo](https://drive.google.com/file/d/1gigNjL6QLCnHk2Gd94whlp35hqMw4NIW/view?usp=drive_link)

## 📊 Dataset Overview
The dataset includes **8,501** examples of metaphors and interpretations.
- **Data Source:** Collected from the Van Nghe Newspaper - Vietnam Writers' Association, covering figurative-rich genres such as poetry, prose, and journalism.
- **Dataset Splitting:**
  - **Train Set:** 5,950 samples.
  - **Dev Set:** 850 samples.
  - **Test Set:** 1,701 samples.
- **Metaphor Classification System:** Consists of 5 main types based on Lakoff & Johnson's theory, with additions customized for Vietnamese characteristics:
  1. Structural Metaphor.
  2. Orientational Metaphor.
  3. Ontological Metaphor.
  4. Emotional Metaphor.
  5. Folklore Metaphor.

## 🎯 Evaluation Tasks (Multitask)
The project designs a multitask processing flow for a comprehensive evaluation of LLMs' capabilities:
- **Task 1A (Identification):** Identify whether a sentence contains a metaphor.
- **Task 1B (Span Extraction):** Accurately extract the boundaries (start/end positions) of a metaphorical phrase.
- **Task 2 (Classification):** Classify the type of metaphor into 5 specific categories.
- **Task 3 (Interpretation):** Interpret the metaphorical sentence into a literal meaning without altering the original message.
- **Task 4 (Judgement):** Score the quality of the interpretation based on a rubric evaluating accuracy, clarity, and naturalness.

## 🔬 Core Experimental Results
- Experiments were conducted on 7 LLMs (around 7-8B parameters scale) under Zero-shot, Few-shot, and Fine-tuning (SFT + LoRA) settings.
- **Leading Model:** Vistral-7B-Chat achieved the best performance due to its optimization for Vietnamese, particularly reaching an F1-score of ~0.772 for the identification task after fine-tuning.
- **Fine-tuning Efficiency:** Fine-tuning significantly reduced "false positive" errors and helped the model conform better to the scoring rubric compared to standard prompting configurations.

## 🗂 Project Structure
- **`Dataset/`**: Contains standard JSON data files for training and evaluation purposes.
- **`Report/`**: Contains the detailed thesis report and presentation slides.
- **`Source/Annotation tool/`**: An annotation tool built on the Django framework.
- **`Source/Demo application/`**: A demo application built with Streamlit demonstrating the processing pipeline.
- **`Source/Experiments with LLMs/`**: Experimental source code tested on models like Llama-3.1, Qwen2.5, Vistral, VinaLLaMA, etc.

## 🚀 Run the Annotation Tool Locally

### 1. Create the environment

```bash
cd "Source/Annotation tool"
python -m venv .venv
pip install -r requirements.txt
```

### 2. Configure credentials

Set the required environment variables before starting Django. Never commit real credentials.

```bash
export DJANGO_SECRET_KEY="replace-with-a-new-random-secret"
export DJANGO_DEBUG="True"
export DB_NAME="ViMUNCH_annotation"
export DB_USER="root"
export DB_PASSWORD="your-local-password"
export DB_HOST="localhost"
export DB_PORT="3306"
```

On Windows PowerShell, use `$env:VARIABLE_NAME="value"` instead of `export`.

Generate a new Django key with:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 3. Initialize and run

```bash
python manage.py migrate
python manage.py runserver
```

> Security: credentials previously committed to Git history must be rotated. Removing them from the current source does not invalidate exposed credentials.

## 🤝 Contribution
For any contributions or requests to use this dataset for research purposes, please contact the authors via the information provided in the report.
