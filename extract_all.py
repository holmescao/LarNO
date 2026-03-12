import fitz

doc = fitz.open('d:/BaiduSyncdisk/code_repository/urbanflood/larFNO/project/Manuscript.pdf')

out_path = 'd:/BaiduSyncdisk/code_repository/urbanflood/larFNO/project/pdf_all_pages.txt'

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(f'Total pages: {doc.page_count}\n\n')
    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text()
        f.write(f'=== Page {i+1} ===\n')
        f.write(text)
        f.write('\n\n')

print('Done')
