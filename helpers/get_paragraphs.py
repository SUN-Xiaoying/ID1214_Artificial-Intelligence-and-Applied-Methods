from docx import Document

def get_paragraphs_text(path):
    """
    param path: word.docx
    return list
    """
    document = Document(path)
    all_paragraphs = document.paragraphs
    paragraphs_text = []
    for paragraph in all_paragraphs:
        # 拼接一个list,包括段落的结构和内容
        paragraphs_text.append([paragraph.style.name,paragraph.text])
    return paragraphs_text
