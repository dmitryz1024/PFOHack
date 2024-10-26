import pandas as pd
from pypdf import PdfReader
from docx import Document
import glob
import os

def read_data(file_path):
    if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError('Unsupported file format')

def read_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return pd.DataFrame([line.split() for line in text.splitlines()])

def read_docx(file_path):
    doc = Document(file_path)
    text = '\n'.join([p.text for p in doc.paragraphs])
    return pd.DataFrame([line.split() for line in text.splitlines()])

def load_all_data(directory):
    all_data = []
    print('Файлы в директории:')
    print(os.listdir(directory))
    for file in glob.glob(f'{directory}/*'):
        try:
            data = read_data(file)
            all_data.append(data)
            print(f'Загружен файл: {file}, размер: {data.shape}')
        except Exception as e:
            print(f'Ошибка при загрузке файла {file}: {e}')
    if not all_data:
        print('Нет данных для объединения')
    return pd.concat(all_data, ignore_index = True) if all_data else pd.DataFrame()

def preprocess_data(data):
    if 'last_deal_date' in data.columns:
        current_date = pd.to_datetime('now')
        data['last_deal_date'] = (current_date - pd.to_datetime(data['last_deal_date'])).dt.days
    return data

def get_client_data(data, client_id):
    client_data = data[data['client_id'] == client_id]
    if client_data.empty:
        raise ValueError(f'Клиент с ID {client_id} не найден')
    return client_data

def get_client_weight(rf_model, feature_extractor, client_data):
    client_features = client_data.drop(columns = ['client_id', 'target'], errors = 'ignore')
    neural_features = feature_extractor.predict(client_features)
    weight = rf_model.predict_proba(neural_features)[:, 1][0] * 100
    return weight