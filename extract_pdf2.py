import fitz
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

doc = fitz.open('d:/BaiduSyncdisk/code_repository/urbanflood/larFNO/project/Manuscript.pdf')

out_path = 'd:/BaiduSyncdisk/code_repository/urbanflood/larFNO/project/pdf_extracted.txt'

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(f'Total pages: {doc.page_count}\n\n')
    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text()
        keywords = ['Table', 'Baseline', 'R2', 'MAE', 'RMSE', 'CSI', 'baseline', 'table', 'Inference', 'Memory', 'GPU', 'MSE', 'FNO', 'SWE', 'UNet', 'ConvLSTM', 'LSTM']
        if any(kw in text for kw in keywords):
            f.write(f'=== Page {i+1} ===\n')
            f.write(text)
            f.write('\n\n')

print('Done')
