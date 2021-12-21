import os
import paddlehub as hub
from helpers import get_paragraphs
from docx import Document

text = get_paragraphs.get_paragraphs_text('docx/zh-demo.docx')

model = hub.Module(name='transformer_zh-en', beam_size=5)
new_document = Document()
for item in text:
    src_style = item[0]
    src_texts = [item[1]]
    print('Heading level: ', src_style)
    print(src_texts)
    n_best = 1  # 每个输入样本的输出候选句子数量
    trg_texts = model.predict(src_texts, n_best=n_best)
    print(trg_texts)
    if src_style == 'Title':
        new_document.add_heading(trg_texts, 0)
    elif src_style[:7:] == 'Heading':
        new_document.add_heading(trg_texts, level=int(src_style[-1]))
    else:
        new_document.add_paragraph(trg_texts)
new_document.save('docx/translate_demo.docx')
