import fitz

doc = fitz.open('d:/BaiduSyncdisk/code_repository/urbanflood/larFNO/project/Manuscript.pdf')
print(f'Total pages: {doc.page_count}')
for i in range(doc.page_count):
    page = doc[i]
    text = page.get_text()
    keywords = ['Table', 'Baseline', 'R2', 'MAE', 'RMSE', 'CSI', 'R\u00b2', 'baseline', 'table', 'Inference', 'Memory', 'GPU', 'MSE']
    if any(kw in text for kw in keywords):
        print(f'=== Page {i+1} ===')
        print(text)
        print()
