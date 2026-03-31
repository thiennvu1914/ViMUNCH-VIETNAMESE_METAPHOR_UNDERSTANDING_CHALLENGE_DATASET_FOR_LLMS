import re
import pandas as pd
from langdetect import detect, LangDetectException
import streamlit as st

# Hàm preprocess_text đã bao gồm split_sentences_custom cải tiến
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    # Giữ nguyên ký tự mong muốn
    text = re.sub(
        r'[^\w\s\.,!\?;:áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ'
        r'ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
        r'"“”‘’\'\–\—…\-\(\)]',
        '', text
    )
    # Xóa khoảng trắng thừa/newline
    text = re.sub(r'\s+', ' ', text).strip()

    # Hàm tách câu cải tiến
    def split_sentences_custom(s):
        sentences = []
        buf = []
        inside_quote = False
        paren_depth = 0
        i = 0
        length = len(s)
        while i < length:
            ch = s[i]
            # Toggle quote
            if ch in ('"', '“', '”', '‘', '’', "'"):
                buf.append(ch)
                inside_quote = not inside_quote
                i += 1
                continue
            # Depth ngoặc tròn
            if ch == '(':
                paren_depth += 1; buf.append(ch); i += 1; continue
            if ch == ')':
                if paren_depth > 0: paren_depth -= 1
                buf.append(ch); i += 1; continue

            # Xử lý chuỗi chấm liên tiếp
            if ch == '.':
                j = i
                while j < length and s[j] == '.':
                    j += 1
                dot_count = j - i
                if dot_count >= 3:
                    # peek next non-space
                    k = j
                    while k < length and s[k].isspace():
                        k += 1
                    next_char = s[k] if k < length else ''
                    if not next_char or next_char.isupper():
                        punct = s[i:j]
                        buf.append(punct)
                        i = j
                        if inside_quote or paren_depth > 0:
                            continue
                        sentence = ''.join(buf).strip()
                        if sentence:
                            sentences.append(sentence)
                        buf = []
                        while i < length and s[i].isspace():
                            i += 1
                        continue
                    else:
                        buf.append(s[i:j])
                        i = j
                        continue
                else:
                    # dot_count <3
                    prev_char = s[i-1] if i-1>=0 else ''
                    next_char = s[i+1] if i+1<length else ''
                    if prev_char.isdigit() and next_char.isdigit():
                        buf.append('.'); i += 1; continue
                    # else tách câu
                    buf.append('.'); i += 1
                    if inside_quote or paren_depth > 0:
                        continue
                    sentence = ''.join(buf).strip()
                    if sentence:
                        sentences.append(sentence)
                    buf = []
                    while i < length and s[i].isspace():
                        i += 1
                    continue

            # Xử lý ellipsis Unicode ‘…’
            if ch == '…':
                j = i + 1
                while j < length and s[j].isspace():
                    j += 1
                next_char = s[j] if j < length else ''
                if not next_char or next_char.isupper():
                    buf.append('…'); i += 1
                    if inside_quote or paren_depth > 0:
                        continue
                    sentence = ''.join(buf).strip()
                    if sentence:
                        sentences.append(sentence)
                    buf = []
                    while i < length and s[i].isspace():
                        i += 1
                    continue
                else:
                    buf.append('…'); i += 1; continue

            # Xử lý dấu ! hoặc ?
            if ch in ('!', '?'):
                buf.append(ch); i += 1
                if inside_quote or paren_depth > 0:
                    continue
                sentence = ''.join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                buf = []
                while i < length and s[i].isspace():
                    i += 1
                continue

            # Bình thường
            buf.append(ch); i += 1

        last = ''.join(buf).strip()
        if last:
            sentences.append(last)
        return sentences

    sents = split_sentences_custom(text)

    # Lọc câu
    valid = []
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        if re.fullmatch(r'[\s\d\.,!\?;:\-"“”‘’\'\–\—…\(\)]+', sent):
            continue
        if len(sent) < 10:
            valid.append(sent)
        else:
            try:
                if detect(sent) == 'vi':
                    valid.append(sent)
            except Exception:
                valid.append(sent)
    return valid

def preprocess_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df2 = df.copy()
    # Ép str để giữ nguyên chuỗi gốc
    df2[text_column] = df2[text_column].fillna('').astype(str)
    df2['processed'] = df2[text_column].apply(preprocess_text)
    return df2.explode('processed').reset_index(drop=True)

# Streamlit UI
st.set_page_config(page_title="Tiền Xử Lý Văn Bản Tiếng Việt", page_icon="📝")
st.title("🛠️ Tiền Xử Lý Văn Bản Tiếng Việt")
st.write("""
- Xóa icon/hashtag/ký tự đặc biệt
- Xóa câu không phải tiếng Việt
- Tách câu và chuẩn hóa
""")

uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
if uploaded_file:
    # Đọc luôn dtype=str để giữ nguyên chuỗi gốc
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
    except Exception:
        # nếu có lỗi encoding, thử đọc mặc định rồi ép str
        df = pd.read_csv(uploaded_file)
        for col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    st.subheader("Xem trước dữ liệu")
    st.write("Kiểu dữ liệu các cột:")
    st.write(df.dtypes)
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("Chọn cột chứa văn bản để xử lý", df.columns)

    # Cho tùy chọn tách theo newline riêng trước khi xử lý toàn bộ chuỗi
    split_by_line = st.checkbox("Xử lý theo từng dòng (split newline trước)", value=False)
    if split_by_line:
        st.info("Nếu bật, mỗi ô sẽ được tách theo newline trước khi tách câu.")

    if st.button("Xử lý dữ liệu"):
        # Nếu select đúng
        # Chuẩn bị DataFrame xử lý
        df_proc = df.copy()
        df_proc[text_col] = df_proc[text_col].fillna('').astype(str)

        # Nếu split_by_line: tái cấu trúc thành nhiều dòng per newline, giữ cột khác
        if split_by_line:
            # Tạo DataFrame exploded theo newline
            rows = []
            for idx, row in df_proc.iterrows():
                text = row[text_col]
                lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
                if lines:
                    for ln in lines:
                        new_row = row.copy()
                        new_row[text_col] = ln
                        rows.append(new_row)
                else:
                    # ô rỗng hoặc chỉ whitespace
                    new_row = row.copy()
                    new_row[text_col] = ''
                    rows.append(new_row)
            df_proc = pd.DataFrame(rows)

        # Áp dụng preprocess
        df_processed = preprocess_df(df_proc, text_col)
        # Kết hợp thêm cột gốc (tùy ý)
        # Ví dụ: nếu split_by_line, có thể thêm cột "original_index" lưu idx gốc.
        st.subheader("Kết quả xử lý")
        # Hiển thị: cột gốc + câu đã xử lý
        # Nếu không split_by_line, cột gốc có thể là toàn bộ văn bản, xem xét hiển thị ngắn
        display_df = df_processed.copy()
        display_df = display_df.rename(columns={text_col: "Văn bản gốc", "processed": "Câu đã xử lý"})
        # Nếu văn bản gốc dài, có thể show truncate
        display_df["Văn bản gốc"] = display_df["Văn bản gốc"].apply(lambda x: x if len(x)<=50 else x[:50]+'...')
        st.dataframe(display_df[["Văn bản gốc", "Câu đã xử lý"]], use_container_width=True)

        st.subheader("Thống kê")
        st.write(f"- Số bản ghi ban đầu: {len(df)}")
        st.write(f"- Số bản ghi sau xử lý (sau explode câu): {len(df_processed)}")

        # Cho tải về
        csv = df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tải kết quả CSV",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv'
        )
