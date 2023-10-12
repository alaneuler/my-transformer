zh_file_path = 'data/zh-en/news-commentary-v13.zh-en.zh'
en_file_path = 'data/zh-en/news-commentary-v13.zh-en.en'

def load_data(start: int, end: int):
    with open(zh_file_path, 'r') as zh_file, open(en_file_path, 'r') as en_file:        
        for i, (zh_line, en_line) in enumerate(zip(zh_file, en_file)):
            if i < start:
                continue
            if i >= end:
                break

            yield zh_line.strip(), en_line.strip()
