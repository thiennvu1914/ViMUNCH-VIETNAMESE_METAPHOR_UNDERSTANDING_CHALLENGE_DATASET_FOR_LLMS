import re
import streamlit as st
from langdetect import detect, LangDetectException
import unicodedata

def is_poetry(text):
    lines = text.strip().split('\n')
    poetic_lines = [line.strip() for line in lines if line.strip() and not re.fullmatch(r'\d+\.?', line.strip())]
    if len(poetic_lines) >= 3:
        short_lines = sum(1 for l in poetic_lines if len(l.split()) <= 8)
        line_break_ratio = len(poetic_lines) / max(1, sum(text.count(p) for p in ['.', '!', '?', '…']))
        return (short_lines / len(poetic_lines) > 0.6 or line_break_ratio > 1.5)
    return False

def merge_short_poetic_lines(lines, min_words=8, max_merge=2):
    merged, buffer, buffer_count = [], "", 0
    for line in lines:
        if len(line.split()) < min_words:
            buffer += ' ' + line
            buffer_count += 1
            if buffer_count >= max_merge:
                merged.append(buffer.strip())
                buffer = ""
                buffer_count = 0
        else:
            if buffer:
                merged.append((buffer + ' ' + line).strip())
                buffer = ''
                buffer_count = 0
            else:
                merged.append(line)
    if buffer:
        merged.append(buffer.strip())
    return merged

def has_latin_char(token):
    """
    Trả về True nếu token chứa ít nhất một ký tự thuộc Unicode script Latin
    (ví dụ chữ A-Z, a-z hoặc chữ Latin có dấu như á, ă, đ, ê,...).
    Nếu token chứa ký tự số thì luôn trả về True.
    Dùng unicodedata.name để kiểm tra 'LATIN' trong tên ký tự.
    """
    for ch in token:
        if ch.isdigit():
            return True
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if 'LATIN' in name:
            return True
    return False

abbrev_list = ["PGS", "TS", "Ths", "GS", "BS", "CN"]  # không chứa dấu chấm; sẽ kiểm tra khi gặp '.'
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
                # Đếm số lượng dấu chấm liên tiếp
                j = i
                while j < length and s[j] == '.':
                    j += 1
                dot_count = j - i
                if dot_count >= 3:
                    # Ellipsis dài; peek next non-space
                    k = j
                    while k < length and s[k].isspace():
                        k += 1
                    next_char = s[k] if k < length else ''
                    if not next_char or next_char.isupper():
                        # Tách câu tại ellipsis dài
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
                        # Không tách: coi ellipsis là phần câu
                        buf.append(s[i:j]); i = j; continue
                else:
                    # dot_count < 3, chỉ một hoặc hai dấu chấm
                    # Trước tiên kiểm tra viết tắt
                    # Lấy đoạn trước dấu chấm, tối đa dài nhất trong abbrev_list
                    matched_abbrev = False
                    for abbr in abbrev_list:
                        L = len(abbr)
                        # kiểm tra s[i-L:i] == abbr (case-sensitive? thường là uppercase)
                        if i - L >= 0 and s[i-L:i] == abbr:
                            # Có thể thêm kiểm tra ranh giới: trước abbr phải là whitespace hoặc bắt đầu chuỗi
                            # Ví dụ: đảm bảo s[i-L-1] là whitespace hoặc không phải chữ cái
                            if i-L-1 < 0 or not s[i-L-1].isalpha():
                                matched_abbrev = True
                                break
                    if matched_abbrev:
                        # Đây là dấu chấm trong viết tắt, giữ nguyên, không tách
                        buf.append('.')
                        i += 1
                        continue

                    # Nếu không phải viết tắt, kiểm tra giữa hai số
                    prev_char = s[i-1] if i-1>=0 else ''
                    next_char = s[i+1] if i+1<length else ''
                    if prev_char.isdigit() and next_char.isdigit():
                        buf.append('.'); i += 1; continue

                    # Không phải viết tắt, không giữa số: tách câu
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
                # Peek next non-space
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

            # Xử lý dấu ‘。’
            if ch == '。':
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

        # Phần dư cuối
        last = ''.join(buf).strip()
        if last:
            sentences.append(last)
        return sentences

# Hàm merge 2 câu khi quoted kết thúc và next bắt đầu bằng comma
def normalize_quote_sentences(text):
    """
    Thêm dấu chấm sau dấu đóng ngoặc kép nếu trước đó có ':' hoặc ','
    và ngay sau không có dấu kết thúc câu; xóa newline giữa.
    Nhận diện cả ASCII " và Unicode quotes “ ”.
    """
    # Pattern: 
    #   ([,:])      nhóm 1: dấu ':' hoặc ','
    #   \s*\n?\s*   optional whitespace/newline
    #   ([“"])\s*   nhóm 2: dấu mở là ASCII " hoặc Unicode “
    #   (.+?)       nhóm 3: nội dung bên trong, non-greedy
    #   ([”"])      nhóm 4: dấu đóng là Unicode ” hoặc ASCII "
    #   (?![.?!…])  ngay sau dấu đóng không có ., ?, !, …
    pattern = re.compile(
        r'([,:])\s*\n?\s*([“"])(.+?)([”"])(?![.?!…])',
        flags=re.DOTALL
    )
    def repl(m):
        punct = m.group(1)         # ':' hoặc ','
        open_q = m.group(2)        # “ hoặc "
        inner = m.group(3)         # nội dung
        close_q = m.group(4)       # ” hoặc "
        # Kết quả: giữ dấu trước, space + mở quote + nội dung + đóng quote + "."
        # Nếu open_q là “ thì close_q có thể là ”; giữ nguyên loại quote
        return f'{punct} {open_q}{inner}{close_q}.'
    prev = None
    out = text
    while prev != out:
        prev = out
        out = pattern.sub(repl, out)
    return out

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    raw = text
    # 1. Xóa mọi cụm số thứ tự dạng "123. " hoặc "1." ở đầu hoặc giữa:
    #    Sử dụng regex \b\d+\.\s* để xóa. 
    #    Lưu ý: có thể xóa ở mọi vị trí.
    # Xóa số thứ tự dạng "123. " hoặc "1." nhưng KHÔNG xóa số thập phân như "1.5"
    # Chỉ xóa nếu sau dấu chấm là khoảng trắng hoặc xuống dòng (không phải số)
    '''raw = re.sub(r'\b\d+\.(?=\s)(?!\d)', '', raw)'''
    # Xóa số thứ tự dạng "123. " hoặc "1. " ở đầu mỗi dòng
    raw = re.sub(r'(?m)^\s*\d+\.\s*', '', raw)
    # Xóa các cụm từ "Dịch nghĩa", "Dịch nghĩa:", "Dịch thơ", "Dịch thơ:"
    raw = re.sub(r'\bDịch (nghĩa|thơ):?\b', '', raw, flags=re.IGNORECASE)

    # 2. Loại bỏ mọi từ không chứa ký tự Latin:
    #    Xử lý theo dòng để giữ newline nếu cần dùng sau.
    new_lines = []
    for ln in raw.split('\n'):
        tokens = ln.split()
        kept = []
        for tok in tokens:
            if has_latin_char(tok):
                kept.append(tok)
            # nếu tok không có ký tự Latin, drop
        if kept:
            new_lines.append(' '.join(kept))
        # nếu kept rỗng → dòng bỏ luôn
    raw = '\n'.join(new_lines)
    # 3. Xóa dòng chỉ chứa số thứ tự nếu còn sót (thực ra bước 1 đã xóa, nhưng lặp lại an toàn)
    raw = '\n'.join([ln for ln in raw.split('\n') if not re.fullmatch(r'\s*\d+\.?\s*', ln.strip())])
    # 4. Làm sạch ký tự đặc biệt không mong muốn, giữ \n
    text_cleaned = re.sub(
        r"[^\w\s\.,!?;:áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồỗổỗơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ"
        r"ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
        r"\"“”‘’\'\–\—…\-\(\)。\n]", '', raw)
    text_cleaned = re.sub(r"[ \t]+", ' ', text_cleaned)
     # 5. Xác định thơ vs văn xuôi, đếm dấu kết thúc câu
    end_puncts = sum(text_cleaned.count(p) for p in ['.', '!', '?', '…'])
    is_poem_flag = is_poetry(raw)
    if is_poem_flag and end_puncts < (len([ln for ln in raw.split('\n') if ln.strip()]) // 2):
        # Nhánh thơ ít dấu: tách theo dòng, merge ngắn
        lines = [ln.strip() for ln in raw.split('\n') if ln.strip()]
        sents = merge_short_poetic_lines(lines)
    else:
        # Nhánh bình thường / thơ có dấu: normalize quotes, nối inline rồi tách dấu
        norm = normalize_quote_sentences(raw)

        inline = re.sub(r"[ \t]+", ' ', norm.replace('\n', ' ')).strip()
        # Xóa câu cuối của inline (nếu có)
        inline = '.'.join(inline.split('.')[:-1]) if '.' in inline else inline
        # Xóa các câu chứa "Ảnh:" hoặc "ảnh:" (không phân biệt hoa thường)
        inline = re.sub(r'[^.!?…]*[Ảả]nh:\s*[^.!?…]*[.!?…]?', '', inline, flags=re.IGNORECASE)
        # Xóa các câu chứa "Nguồn:" hoặc "nguồn:" (không phân biệt hoa thường)
        inline = re.sub(r'[^.!?…]*[Nn]guồn:\s*[^.!?…]*[.!?…]?', '', inline, flags=re.IGNORECASE)
        # Xóa các câu hoặc dòng bắt đầu bằng ">>"
        inline = re.sub(r'(?:^|\n)\s*>>[^\n.!?…]*[.!?…]?', '', inline)
        sents = split_sentences_custom(inline)
        # Nếu cần merge quoted+comma, gọi merge_quoted_with_comma(sents)
        # sents = merge_quoted_with_comma(sents)
    # 6. Hậu xử lý: lọc rác, loại trùng, giới hạn độ dài
    valid = []
    seen = set()
    for sent in sents:
        s = sent.strip().replace('\n', ' ').replace('\r', ' ')
        if not s:
            continue
        # Bỏ câu chỉ ký tự đặc biệt/số
        if re.fullmatch(r"[\s\d\.,!?;:\-\"“”‘’'\(\)…。]+", s):
            continue
        # Giới hạn độ dài: >500 từ thì bỏ
        if len(s.split()) > 500:
            continue
        if s not in seen:
            if len(s) < 10:
                valid.append(s)
                seen.add(s)
            else:
                try:
                    if detect(s) == 'vi':
                        valid.append(s)
                        seen.add(s)
                except:
                    valid.append(s)
                    seen.add(s)
    return valid


# Giao diện Streamlit
st.set_page_config(page_title="Tiền Xử Lý Văn Bản Tiếng Việt", page_icon="📝")
st.title("🛠️ Tiền Xử Lý Văn Bản Tiếng Việt")
st.write("""
- Xóa icon/hashtag/ký tự đặc biệt
- Xóa câu không phải tiếng Việt
- Tách câu và chuẩn hóa
""")

# Chia layout thành 2 cột
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nhập văn bản")
    user_input = st.text_area("Dán văn bản của bạn vào đây:", 
                             height=300,
                             placeholder="Ví dụ: Trên mảnh đất 50 000 người đã chết...\nCó phải em về đêm nay...")

with col2:
    st.subheader("Kết quả xử lý")
    
    if st.button("Xử lý văn bản", use_container_width=True):
        if user_input.strip():
            # Xử lý toàn bộ văn bản, không tách từng dòng
            processed = preprocess_text(user_input.strip())
            results = []

            for idx, sentence in enumerate(processed, start=1):
                results.append({
                    "Câu đã xử lý": f"{idx}. {sentence}",
                    "Số câu": len(processed)
                })
                
                if results:
                    # Hiển thị kết quả trong bảng
                    st.dataframe([r["Câu đã xử lý"] for r in results], 
                                use_container_width=True,
                                column_config={"value": "Câu đã xử lý"})
                else:
                    st.warning("Không tìm thấy câu tiếng Việt hợp lệ nào sau khi xử lý")
            else:
                st.error("Vui lòng nhập văn bản cần xử lý")

# Ví dụ mẫu
with st.expander("📚 Ví dụ mẫu (Bấm để xem)"):
    st.code("""
Kiến nghị Chính phủ, 27 nhà đầu tư trong và ng...
Trên mảnh đất 50 000 người đã chết\nKhông ai ...
Có phải em về đêm nay\nTrên con đường thời gia...
Khu đô thị Cù Lao Bến Đình được quy hoạch là k...
Quái dữ a!\nThế sự gẫm ngán trân, người trong ...
PhápDiễn viên Thúy Ngân lần đầu diễn thời tran...
MediaTek cho biết muốn "phổ cập AI" từ thiết b...
ItalyTay vợt số một thế giới Jannik Sinner vào...
黃\n山\n日\n記\n其\n六\n地\n方\n同\n志\n太\n客\n氣\n，\n對\n我...
Hóa chất "nước kẹo" - tên khoa học là 6-Benzyl...
    """.strip())

# Footer
st.divider()
st.caption("Ứng dụng tiền xử lý văn bản tiếng Việt | Sử dụng thư viện langdetect để nhận diện ngôn ngữ")