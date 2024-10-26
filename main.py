from pickle import load
import os
try:
    from getting_DataFrame import *
    from creating_DataSet import *
except ImportError:
    os.system('pip install --upgrade pip > nul')
    os.system('pip install pandas click lxml xlrd openpyxl python-docx pypdf scikit-learn tensorflow > nul')
    os.system('cls')

def main():
    create_dataset(list(glob.glob('meta/*')), 'test_input', 0)
    data = load_all_data('test_input')
    data = preprocess_data(data)
    model, feature_extractor, rf_model = load(open('dndt_model.sav', 'rb'))
    client_weights = {}
    for client_id in data['client_id'].unique():
        client_data = get_client_data(data, client_id)
        weight = get_client_weight(rf_model, feature_extractor, client_data)
        client_weights[client_id] = weight
    sorted_clients = sorted(client_weights.items(), key = lambda x: x[1], reverse = True)
    print("Ранжированный список клиентов:", sorted_clients)

if __name__ == "__main__":
    main()