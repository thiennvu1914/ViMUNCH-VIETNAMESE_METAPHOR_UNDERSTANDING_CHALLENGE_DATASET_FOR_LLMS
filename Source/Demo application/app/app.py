"""
ViMUNCH Demo - Vietnamese Metaphor Analysis
Streamlit Application

Pipeline:
- Task 1a, 1b, 2, 4: Vistral Zero-shot / Fine-tuning
- Task 3: Vistral Few-shot 5
"""
import streamlit as st
import json
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    TRAIN_PATH,
    DEV_PATH,
    TEST_PATH,
    FEWSHOT_IDS,
    ALLOWED_TYPES,
    TYPE_DESCRIPTIONS,
    FT_MODEL_PATH,
)
from utils.model_loader import (
    get_model_and_tokenizer,
    get_model_info,
    check_gpu_available,
)
from utils.inference import run_full_pipeline, run_task_4

# Bản đồ tên tiếng Việt cho loại ẩn dụ
METAPHOR_TYPE_NAMES = {
    "structural": "Ẩn dụ cấu trúc",
    "ontological": "Ẩn dụ bản thể",
    "orientational": "Ẩn dụ định hướng",
    "emotional": "Ẩn dụ cảm xúc",
    "cultural_folklore": "Ẩn dụ văn hóa dân gian"
}

# Helper function để in ra stderr (không bị Streamlit buffer)
def log(msg):
    import sys
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="ViMUNCH - Vietnamese Metaphor Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-container {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #6366F1;
    }
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
    }
    .task-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .task-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
    }
    .task-icon-detect { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .task-icon-extract { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
    .task-icon-classify { background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); }
    .task-icon-interpret { background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%); }
    .task-icon-score { background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%); }
    .task-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
    }
    .task-subtitle {
        font-size: 0.8rem;
        color: #64748B;
    }
    .task-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 500;
        margin-left: auto;
    }
    .badge-success { background: #D1FAE5; color: #059669; }
    .badge-info { background: #DBEAFE; color: #2563EB; }
    .badge-warning { background: #FEF3C7; color: #D97706; }
    .sentence-display {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 12px;
        padding: 1.25rem;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #334155;
        border: 1px solid #E2E8F0;
    }
    .interpretation-box {
        background: linear-gradient(135deg, #FDF4FF 0%, #FAF5FF 100%);
        border-radius: 12px;
        padding: 1.25rem;
        font-size: 1rem;
        line-height: 1.7;
        color: #581C87;
        border: 1px solid #E9D5FF;
        font-style: italic;
    }
    .metaphor-highlight {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        color: #92400E;
        border: 1px solid #F59E0B;
    }
    .type-tag {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px 4px 2px 0;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .no-metaphor-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #BBF7D0;
    }
    .no-metaphor-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .no-metaphor-text {
        font-size: 1.1rem;
        color: #166534;
        font-weight: 500;
    }
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ====== HELPER FUNCTIONS ======
@st.cache_data
def load_dataset(path: str):
    """Load dataset from JSON file"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_all_datasets():
    """Load all datasets"""
    train = load_dataset(TRAIN_PATH)
    dev = load_dataset(DEV_PATH)
    test = load_dataset(TEST_PATH)
    return train, dev, test


@st.cache_data
def get_fewshot_examples(_train_data, fewshot_ids):
    """Get fixed few-shot examples"""
    by_str = {str(r["id"]): r for r in _train_data}
    few = []
    for _id in fewshot_ids:
        r = by_str.get(str(_id))
        if r is not None:
            few.append(r)
    return few


@st.cache_resource(show_spinner="🔄 Đang tải model Vistral-7B-Chat... (lần đầu sẽ mất vài phút)")
def load_models(use_finetuned: bool):
    """Load models (cached by Streamlit)"""
    log("📦 Loading Base Model...")
    model_base, tokenizer_base = get_model_and_tokenizer(use_finetuned=False)
    log("✅ Base model loaded!")
    
    model_ft, tokenizer_ft = None, None
    if use_finetuned and os.path.exists(FT_MODEL_PATH):
        log("📦 Loading Fine-tuned Model...")
        model_ft, tokenizer_ft = get_model_and_tokenizer(use_finetuned=True)
        log("✅ Fine-tuned model loaded!")
    
    return model_base, tokenizer_base, model_ft, tokenizer_ft


def highlight_spans(sentence: str, spans: list) -> str:
    """Highlight metaphor spans"""
    if not spans:
        return sentence
    sorted_spans = sorted(spans, key=lambda x: x["start"], reverse=True)
    result = sentence
    for sp in sorted_spans:
        start, end = sp["start"], sp["end"]
        phrase = sp["phrase"]
        if 0 <= start < end <= len(result) and result[start:end] == phrase:
            highlighted = f'<span class="metaphor-highlight">{phrase}</span>'
            result = result[:start] + highlighted + result[end:]
    return result


def create_type_distribution_chart(data: list):
    """Create metaphor type distribution chart"""
    type_counts = Counter()
    for item in data:
        if item.get("have_metaphor") == 1:
            for t in (item.get("metaphor_types") or []):
                type_counts[t] += 1
    if not type_counts:
        return None
    df = pd.DataFrame([{"Type": t, "Count": c} for t, c in type_counts.most_common()])
    fig = px.bar(df, x="Type", y="Count", color="Count",
                 color_continuous_scale=["#818CF8", "#6366F1", "#4F46E5"])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), showlegend=False, coloraxis_showscale=False,
        xaxis_title="", yaxis_title="Số lượng", margin=dict(t=20, b=20)
    )
    return fig


def create_metaphor_ratio_chart(data: list):
    """Create metaphor ratio pie chart"""
    has_metaphor = sum(1 for item in data if item.get("have_metaphor") == 1)
    no_metaphor = len(data) - has_metaphor
    fig = go.Figure(data=[go.Pie(
        labels=["Có ẩn dụ", "Không có ẩn dụ"],
        values=[has_metaphor, no_metaphor],
        hole=0.6,
        marker_colors=["#6366F1", "#E2E8F0"],
        textinfo="percent+value"
    )])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(t=20, b=40)
    )
    return fig


def create_sentence_length_chart(data: list):
    """Create sentence length histogram"""
    lengths = [len(item.get("sentence", "")) for item in data]
    fig = px.histogram(x=lengths, nbins=30, color_discrete_sequence=["#8B5CF6"])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), xaxis_title="Độ dài câu (ký tự)",
        yaxis_title="Số lượng", margin=dict(t=20, b=20)
    )
    return fig


def create_types_per_sentence_chart(data: list):
    """Create distribution of number of metaphor types per sentence"""
    type_counts = []
    for item in data:
        if item.get("have_metaphor") == 1:
            num_types = len(item.get("metaphor_types") or [])
            type_counts.append(num_types)
    if not type_counts:
        return None
    df = pd.DataFrame({"Số loại ẩn dụ": type_counts})
    fig = px.histogram(df, x="Số loại ẩn dụ", nbins=5, color_discrete_sequence=["#EC4899"])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), xaxis_title="Số loại ẩn dụ/câu",
        yaxis_title="Số câu", margin=dict(t=20, b=20)
    )
    return fig


def create_spans_per_sentence_chart(data: list):
    """Create distribution of number of spans per sentence"""
    span_counts = []
    for item in data:
        if item.get("have_metaphor") == 1:
            num_spans = len(item.get("metaphor_phrases") or [])
            span_counts.append(num_spans)
    if not span_counts:
        return None
    df = pd.DataFrame({"Số span": span_counts})
    fig = px.histogram(df, x="Số span", nbins=10, color_discrete_sequence=["#10B981"])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), xaxis_title="Số span ẩn dụ/câu",
        yaxis_title="Số câu", margin=dict(t=20, b=20)
    )
    return fig


def create_span_length_chart(data: list):
    """Create distribution of span lengths (in characters)"""
    span_lengths = []
    for item in data:
        if item.get("have_metaphor") == 1:
            for sp in (item.get("metaphor_phrases") or []):
                phrase = sp.get("phrase", "")
                if phrase:
                    span_lengths.append(len(phrase))
    if not span_lengths:
        return None
    fig = px.histogram(x=span_lengths, nbins=25, color_discrete_sequence=["#F59E0B"])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), xaxis_title="Độ dài span (ký tự)",
        yaxis_title="Số lượng", margin=dict(t=20, b=20)
    )
    return fig


def create_type_cooccurrence_chart(data: list):
    """Create type co-occurrence heatmap"""
    types = ALLOWED_TYPES
    cooccurrence = {t1: {t2: 0 for t2 in types} for t1 in types}
    
    for item in data:
        if item.get("have_metaphor") == 1:
            item_types = item.get("metaphor_types") or []
            for t1 in item_types:
                for t2 in item_types:
                    if t1 in cooccurrence and t2 in cooccurrence[t1]:
                        cooccurrence[t1][t2] += 1
    
    matrix = [[cooccurrence[t1][t2] for t2 in types] for t1 in types]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=types,
        y=types,
        colorscale=[[0, "#F8FAFC"], [0.5, "#818CF8"], [1, "#4F46E5"]],
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B", size=9),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(tickangle=45)
    )
    return fig


def create_word_count_chart(data: list):
    """Create word count distribution"""
    word_counts = [len(item.get("sentence", "").split()) for item in data]
    fig = px.histogram(x=word_counts, nbins=25, color_discrete_sequence=["#06B6D4"])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), xaxis_title="Số từ/câu",
        yaxis_title="Số lượng", margin=dict(t=20, b=20)
    )
    return fig


def create_type_by_split_chart(train_data, dev_data, test_data):
    """Create type distribution comparison across splits"""
    data_list = []
    for split_name, data in [("Train", train_data), ("Dev", dev_data), ("Test", test_data)]:
        type_counts = Counter()
        for item in data:
            if item.get("have_metaphor") == 1:
                for t in (item.get("metaphor_types") or []):
                    type_counts[t] += 1
        for t, c in type_counts.items():
            data_list.append({"Split": split_name, "Type": t, "Count": c})
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list)
    fig = px.bar(df, x="Type", y="Count", color="Split", barmode="group",
                 color_discrete_map={"Train": "#6366F1", "Dev": "#EC4899", "Test": "#10B981"})
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B"), xaxis_title="", yaxis_title="Số lượng",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def render_scores_grid(scores: dict):
    """Render scores"""
    if not scores:
        st.info("Không có điểm đánh giá")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Chất lượng diễn giải**")
        for key in ["accuracy", "clarity", "naturalness"]:
            val = scores.get(key, "N/A")
            st.markdown(f"- {key.capitalize()}: **{val}**/4")
        st.markdown(f"**→ Overall: {scores.get('overall', 'N/A')}/4**")
    with col2:
        st.markdown("**Tương đồng ngữ nghĩa**")
        for key in ["meaning", "implication", "modality", "syntax", "context"]:
            val = scores.get(key, "N/A")
            st.markdown(f"- {key.capitalize()}: **{val}**/4")
        st.markdown(f"**→ Quality: {scores.get('quality', 'N/A')}/4**")


def render_analysis_results(sentence: str, result: dict, show_task_4: bool = False, gold_scores: dict = None):
    """Render kết quả phân tích với giao diện chuyên nghiệp"""
    
    have_metaphor = result["task_1a"]["have_metaphor"]
    
    # Task 1a - Nhận diện
    st.markdown("""
    <div class="result-card">
        <div class="task-header">
            <div class="task-icon task-icon-detect">🔍</div>
            <div>
                <div class="task-title">Task 1a: Nhận diện ẩn dụ</div>
                <div class="task-subtitle">Xác định câu có chứa ẩn dụ hay không</div>
            </div>
            <span class="task-badge badge-success">Zero-shot/FT</span>
        </div>
    """, unsafe_allow_html=True)
    
    if have_metaphor == 1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #059669;">
            <span style="font-size: 1.5rem;">✅</span>
            <span style="font-size: 1.1rem; font-weight: 600;">Câu có chứa ẩn dụ</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-metaphor-box">
            <div class="no-metaphor-icon">📝</div>
            <div class="no-metaphor-text">Câu không chứa ẩn dụ</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if have_metaphor == 1:
        # Task 1b - Trích xuất span
        spans = result["task_1b"]["metaphor_phrases"]
        st.markdown("""
        <div class="result-card">
            <div class="task-header">
                <div class="task-icon task-icon-extract">📍</div>
                <div>
                    <div class="task-title">Task 1b: Trích xuất span</div>
                    <div class="task-subtitle">Xác định vị trí các cụm từ ẩn dụ trong câu</div>
                </div>
                <span class="task-badge badge-warning">Zero-shot/FT</span>
            </div>
        """, unsafe_allow_html=True)
        
        if spans:
            highlighted = highlight_spans(sentence, spans)
            st.markdown(f'<div class="sentence-display">{highlighted}</div>', unsafe_allow_html=True)
            
            # Hiển thị danh sách spans
            st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
            for i, sp in enumerate(spans):
                st.markdown(f"""
                <span style="display: inline-block; background: #FEF3C7; padding: 4px 12px; 
                border-radius: 8px; margin: 4px; font-size: 0.9rem; border: 1px solid #F59E0B;">
                    <strong>{i+1}.</strong> "{sp['phrase']}"
                </span>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("Không tìm thấy span ẩn dụ")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Task 2 - Phân loại
        types = result["task_2"]["metaphor_types"]
        st.markdown("""
        <div class="result-card">
            <div class="task-header">
                <div class="task-icon task-icon-classify">🏷️</div>
                <div>
                    <div class="task-title">Task 2: Phân loại ẩn dụ</div>
                    <div class="task-subtitle">Xác định loại ẩn dụ (multi-label)</div>
                </div>
                <span class="task-badge badge-info">Zero-shot/FT</span>
            </div>
        """, unsafe_allow_html=True)
        
        if types:
            type_html = " ".join([f'<span class="type-tag">{t}</span>' for t in types])
            st.markdown(f'<div style="margin-top: 0.5rem;">{type_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("Không phân loại được")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Task 3 - Diễn giải
        interpretation = result["task_3"]["interpretation"]
        st.markdown("""
        <div class="result-card">
            <div class="task-header">
                <div class="task-icon task-icon-interpret">💡</div>
                <div>
                    <div class="task-title">Task 3: Diễn giải ẩn dụ</div>
                    <div class="task-subtitle">Giải thích ý nghĩa của ẩn dụ trong ngữ cảnh</div>
                </div>
                <span class="task-badge badge-success">Few-shot 5</span>
            </div>
        """, unsafe_allow_html=True)
        
        if interpretation:
            st.markdown(f'<div class="interpretation-box"><strong>"{interpretation}"</strong></div>', unsafe_allow_html=True)
        else:
            st.caption("Không tạo được diễn giải")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Task 4 - Chấm điểm (chỉ hiển thị khi show_task_4=True)
        if show_task_4 and interpretation:
            scores = result.get("task_4", {}).get("scores")
            st.markdown("""
            <div class="result-card">
                <div class="task-header">
                    <div class="task-icon task-icon-score">⭐</div>
                    <div>
                        <div class="task-title">Task 4: Chấm điểm diễn giải</div>
                        <div class="task-subtitle">Đánh giá chất lượng diễn giải</div>
                    </div>
                    <span class="task-badge badge-info">Zero-shot/FT</span>
                </div>
            """, unsafe_allow_html=True)
            
            if scores:
                # Bản đồ tên tiếng Việt
                metric_names = {
                    "accuracy": "Độ chính xác",
                    "clarity": "Độ rõ ràng",
                    "naturalness": "Độ tự nhiên",
                    "meaning": "Nghĩa mệnh đề",
                    "modality": "Nghĩa tình thái",
                    "implication": "Hàm ý",
                    "syntax": "Cấu trúc cú pháp",
                    "context": "Ngữ cảnh sử dụng",
                    "overall": "Tổng thể",
                    "quality": "Tổng thể"
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Chất lượng diễn giải**")
                    for key in ["accuracy", "clarity", "naturalness"]:
                        val = scores.get(key, "N/A")
                        progress_val = int(val) / 4 if isinstance(val, (int, float)) else 0
                        st.progress(progress_val, text=f"{metric_names[key]}: {val}/4")
                    overall = scores.get('overall', 'N/A')
                    st.metric(metric_names["overall"], f"{overall}/4")
                
                with col2:
                    st.markdown("**📈 Tương đồng ngữ nghĩa**")
                    for key in ["meaning", "modality", "implication", "syntax", "context"]:
                        val = scores.get(key, "N/A")
                        progress_val = int(val) / 4 if isinstance(val, (int, float)) else 0
                        st.progress(progress_val, text=f"{metric_names[key]}: {val}/4")
                    quality = scores.get('quality', 'N/A')
                    st.metric(metric_names["quality"], f"{quality}/4")
                
                # So sánh với ground truth nếu có
                if gold_scores:
                    st.markdown("---")
                    st.markdown("**📋 So sánh với điểm gốc (Ground Truth)**")
                    compare_df = pd.DataFrame([
                        {"Metric": "Overall", "Ground Truth": gold_scores.get("overall", "N/A"), "Model": scores.get("overall", "N/A")},
                        {"Metric": "Quality", "Ground Truth": gold_scores.get("quality", "N/A"), "Model": scores.get("quality", "N/A")}
                    ])
                    st.dataframe(compare_df, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có điểm đánh giá")
            
            st.markdown("</div>", unsafe_allow_html=True)


# ====== SIDEBAR ======
with st.sidebar:
    st.markdown("### ◈ ViMUNCH")
    st.caption("Vietnamese Metaphor Understanding Challenge")
    st.markdown("---")
    
    st.markdown('<p class="sidebar-title">Trạng thái hệ thống</p>', unsafe_allow_html=True)
    gpu_available, gpu_info = check_gpu_available()
    if gpu_available:
        st.success("GPU: Sẵn sàng")
        st.caption(gpu_info)
    else:
        st.warning("CPU Mode")
        st.caption(gpu_info)
    
    st.markdown("---")
    st.markdown('<p class="sidebar-title">Cấu hình Model</p>', unsafe_allow_html=True)
    ft_exists = os.path.exists(FT_MODEL_PATH)
    use_finetuned = st.toggle(
        "Sử dụng Fine-tuned Model",
        value=ft_exists,
        disabled=not ft_exists,
        help="Dùng model đã fine-tune cho Task 1a, 1b, 2, 4"
    )
    if not ft_exists:
        st.caption("Fine-tuned model chưa được cài đặt")
    
    st.markdown("---")
    st.markdown('<p class="sidebar-title">Pipeline</p>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Task": ["1a", "1b", "2", "3", "4"],
            "Mô tả": ["Nhận diện", "Trích xuất", "Phân loại", "Diễn giải", "Chấm điểm"],
            "Mode": ["ZS/FT", "ZS/FT", "ZS/FT", "FS-5", "ZS/FT"]
        }),
        hide_index=True, use_container_width=True
    )
    st.caption("ZS: Zero-shot | FT: Fine-tuned | FS-5: Few-shot 5")


# ====== MAIN CONTENT ======
st.markdown('<h1 class="main-title">Vietnamese Metaphor Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Demo phân tích ẩn dụ tiếng Việt sử dụng Vistral-7B-Chat</p>', unsafe_allow_html=True)

# Load data
train_data, dev_data, test_data = load_all_datasets()
fewshot_examples = get_fewshot_examples(train_data, FEWSHOT_IDS) if train_data else []

# Load model sử dụng st.cache_resource (model đã được pre-load bởi run.py nên sẽ lấy từ cache)
model_base, tokenizer_base, model_ft, tokenizer_ft = load_models(use_finetuned)

# Hiển thị trạng thái
st.sidebar.success("✅ Model đã sẵn sàng!")

# Main tabs
tab_dashboard, tab_analysis = st.tabs(["Dashboard", "Phân tích"])


# ====== TAB 1: DASHBOARD ======
with tab_dashboard:
    st.markdown("### Tổng quan dữ liệu ViMUNCH")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Set", f"{len(train_data):,}")
    with col2:
        st.metric("Dev Set", f"{len(dev_data):,}")
    with col3:
        st.metric("Test Set", f"{len(test_data):,}")
    with col4:
        total = len(train_data) + len(dev_data) + len(test_data)
        st.metric("Tổng cộng", f"{total:,}")
    
    all_data = train_data + dev_data + test_data
    
    st.markdown("---")
    st.markdown("### 📊 Phân tích tổng quan")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Phân bố loại ẩn dụ")
        fig = create_type_distribution_chart(all_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Tỷ lệ câu có ẩn dụ")
        fig = create_metaphor_ratio_chart(all_data)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Phân bố độ dài câu (ký tự)")
        fig = create_sentence_length_chart(all_data)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Phân bố số từ/câu")
        fig = create_word_count_chart(all_data)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Phân tích chi tiết ẩn dụ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Số loại ẩn dụ/câu")
        fig = create_types_per_sentence_chart(all_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu")
    with col2:
        st.markdown("#### Số span ẩn dụ/câu")
        fig = create_spans_per_sentence_chart(all_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Độ dài span ẩn dụ")
        fig = create_span_length_chart(all_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu")
    with col2:
        st.markdown("#### Ma trận đồng xuất hiện loại ẩn dụ")
        fig = create_type_cooccurrence_chart(all_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu")
    
    st.markdown("---")
    st.markdown("### 📋 So sánh giữa các tập dữ liệu")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Phân bố loại ẩn dụ theo tập")
        fig = create_type_by_split_chart(train_data, dev_data, test_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu")
    with col2:
        st.markdown("#### Thống kê tổng hợp")
        stats = []
        for name, data in [("Train", train_data), ("Dev", dev_data), ("Test", test_data)]:
            has_meta = sum(1 for item in data if item.get("have_metaphor") == 1)
            total_spans = sum(len(item.get("metaphor_phrases") or []) for item in data if item.get("have_metaphor") == 1)
            avg_len = sum(len(item.get("sentence", "")) for item in data) / len(data) if data else 0
            stats.append({
                "Tập": name, 
                "Tổng câu": len(data), 
                "Có ẩn dụ": has_meta,
                "Tỷ lệ %": f"{has_meta/len(data)*100:.1f}" if data else "0",
                "Tổng span": total_spans,
                "TB ký tự/câu": f"{avg_len:.0f}"
            })
        st.dataframe(pd.DataFrame(stats), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📖 Mô tả các loại ẩn dụ")
    cols = st.columns(len(TYPE_DESCRIPTIONS))
    for i, (t, desc) in enumerate(TYPE_DESCRIPTIONS.items()):
        with cols[i]:
            st.markdown(f"**{t}**")
            st.caption(desc)


# ====== TAB 2: ANALYSIS ======
with tab_analysis:
    st.markdown("### Phân tích ẩn dụ")
    
    analysis_mode = st.radio(
        "Chọn chế độ:",
        ["Nhập câu mới", "Chọn từ tập Test"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if analysis_mode == "Nhập câu mới":
        sentence = st.text_area(
            "Nhập câu cần phân tích:",
            placeholder="Ví dụ: UIT là ngôi nhà thứ hai của chúng tôi...",
            height=100
        )
        
        analyze_btn = st.button("Phân tích (Task 1→3)", type="primary")
        
        if analyze_btn and sentence.strip():
            if not fewshot_examples:
                st.error("Không tìm thấy few-shot examples.")
                st.stop()
            
            progress = st.progress(0)
            status = st.empty()
            
            try:
                progress.progress(10)
                status.text("🔄 Task 1a: Nhận diện ẩn dụ...")
                
                result = run_full_pipeline(
                    model_base=model_base,
                    tokenizer_base=tokenizer_base,
                    sentence=sentence.strip(),
                    fewshot_examples=fewshot_examples,
                    model_ft=model_ft,
                    tokenizer_ft=tokenizer_ft,
                    use_ft_for_task_1_2=use_finetuned,
                    use_ft_for_task_4=use_finetuned,
                    skip_task_4=True,
                    progress_callback=lambda m: status.text(m)
                )
                
                progress.progress(100)
                status.empty()
                progress.empty()
                
                st.markdown("---")
                st.markdown("### 📊 Kết quả phân tích")
                
                # Render kết quả đẹp - KHÔNG có Task 4 cho mode nhập câu mới
                render_analysis_results(sentence.strip(), result, show_task_4=False)
                
                with st.expander("Cấu trúc OUTPUT"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Lỗi: {e}")
    
    else:
        # Mode 2: Select from test
        if not test_data:
            st.warning("Không tìm thấy dữ liệu test")
            st.stop()
        
        # Bộ lọc chính
        col1, col2 = st.columns(2)
        with col1:
            filter_metaphor = st.selectbox("Lọc ẩn dụ:", ["Tất cả", "Có ẩn dụ", "Không có ẩn dụ"])
        
        # Chỉ hiển thị lọc loại và diễn giải nếu không chọn "Không có ẩn dụ"
        if filter_metaphor != "Không có ẩn dụ":
            with col2:
                filter_type = st.selectbox("Lọc loại:", ["Tất cả"] + [METAPHOR_TYPE_NAMES[t] for t in ALLOWED_TYPES])
            
            col3, col4, col5 = st.columns(3)
            with col3:
                has_interp = st.selectbox("Có diễn giải:", ["Tất cả", "Có", "Không"])
            with col4:
                score_type = st.selectbox("Loại điểm:", ["Tất cả", "Độ tương quan (overall)", "Chất lượng (quality)"])
            with col5:
                if score_type != "Tất cả":
                    filter_score = st.selectbox("Lọc điểm:", ["≥ 3.5", "≥ 3.0", "< 2.5", "≤ 2.0"])
                else:
                    filter_score = None
        else:
            filter_type = "Tất cả"
            has_interp = "Tất cả"
            score_type = "Tất cả"
            filter_score = None
        
        filtered = test_data
        if filter_metaphor == "Có ẩn dụ":
            filtered = [r for r in filtered if r.get("have_metaphor", 0) == 1]
        elif filter_metaphor == "Không có ẩn dụ":
            filtered = [r for r in filtered if r.get("have_metaphor", 0) == 0]
        
        if filter_type != "Tất cả":
            # Đổi tên tiếng Việt về tiếng Anh để lọc
            type_name_reverse = {v: k for k, v in METAPHOR_TYPE_NAMES.items()}
            english_type = type_name_reverse.get(filter_type)
            if english_type:
                filtered = [r for r in filtered if english_type in (r.get("metaphor_types") or [])]
        
        if has_interp == "Có":
            filtered = [r for r in filtered if r.get("interpretation")]
        elif has_interp == "Không":
            filtered = [r for r in filtered if not r.get("interpretation")]
        
        if score_type != "Tất cả" and filter_score:
            # Xác định loại điểm
            score_key = "overall" if "overall" in score_type else "quality"
            
            # Xử lý ngưỡng điểm
            if filter_score == "≥ 3.5":
                filtered = [r for r in filtered if r.get("scores") and r["scores"].get(score_key, 0) >= 3.5]
            elif filter_score == "≥ 3.0":
                filtered = [r for r in filtered if r.get("scores") and r["scores"].get(score_key, 0) >= 3.0]
            elif filter_score == "< 2.5":
                filtered = [r for r in filtered if r.get("scores") and r["scores"].get(score_key, 0) < 2.5]
            elif filter_score == "≤ 2.0":
                filtered = [r for r in filtered if r.get("scores") and r["scores"].get(score_key, 0) <= 2.0]
        
        st.caption(f"Tìm thấy {len(filtered)} câu")
        
        if filtered:
            sample_idx = st.slider("Chọn mẫu:", 0, len(filtered) - 1, 0)
            sample = filtered[sample_idx]
            sentence = sample.get('sentence', '')
            
            st.markdown("---")
            st.markdown(f"**Câu:** {sentence}")
            st.caption(f"ID: {sample.get('id', 'N/A')}")
            
            # Ground truth section
            with st.expander("Ground Truth (Nhãn gốc)", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    has_meta = sample.get("have_metaphor", 0) == 1
                    st.markdown(f"**Có ẩn dụ:** {'Có' if has_meta else 'Không'}")
                    if has_meta:
                        types = sample.get("metaphor_types", [])
                        if types:
                            st.markdown(" ".join([f'<span class="type-tag">{t}</span>' for t in types]), unsafe_allow_html=True)
                        spans = sample.get("metaphor_phrases", [])
                        if spans:
                            st.markdown("**Spans:**")
                            for sp in spans:
                                st.caption(f"• \"{sp.get('phrase', '')}\"")
                with col2:
                    gold_interp = sample.get("interpretation", "")
                    if gold_interp:
                        st.markdown("**Diễn giải gốc:**")
                        st.markdown(f"> {gold_interp}")
                    gold_scores = sample.get("scores")
                    if gold_scores:
                        st.markdown(f"**Điểm gốc:** Overall={gold_scores.get('overall')} | Quality={gold_scores.get('quality')}")
            
            # Run full pipeline button
            st.markdown("---")
            if st.button("Phân tích (Task 1 → 4)", type="primary", key="run_test_pipeline"):
                if not fewshot_examples:
                    st.error("Không tìm thấy few-shot examples.")
                    st.stop()
                
                progress = st.progress(0)
                status = st.empty()
                
                try:
                    progress.progress(10)
                    status.text("Đang phân tích...")
                    
                    result = run_full_pipeline(
                        model_base=model_base,
                        tokenizer_base=tokenizer_base,
                        sentence=sentence,
                        fewshot_examples=fewshot_examples,
                        model_ft=model_ft,
                        tokenizer_ft=tokenizer_ft,
                        use_ft_for_task_1_2=use_finetuned,
                        use_ft_for_task_4=use_finetuned,
                        progress_callback=lambda m: status.text(m)
                    )
                    
                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    
                    # Render kết quả đẹp - CÓ Task 4 cho mode chọn từ test
                    gold_scores = sample.get("scores")
                    render_analysis_results(sentence, result, show_task_4=True, gold_scores=gold_scores)
                    
                    with st.expander("Cấu trúc OUTPUT"):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"Lỗi: {e}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.875rem;'>"
    "ViMUNCH Demo • Built with Streamlit</div>",
    unsafe_allow_html=True
)
