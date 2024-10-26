from tensorflow import keras
from keras import models, layers, optimizers, regularizers, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import dump
from getting_DataFrame import *

def build_neural_network(input_dim):
    # Принимаем на вход факторы из таблицы
    inputs = layers.Input(shape = (input_dim,))
    # Создаем входной слой с L2 регуляризацией
    x = layers.Dense(128, activation = 'elu', kernel_regularizer = regularizers.l2(0.01))(inputs)
    # Dropout (убираем из процесса обучения неактивные/бесполезные нейроны, 50% от общего числа; регулируется первым аргументом функции)
    x = layers.Dropout(0.5)(x)
    # Создаем скрытый слой с L2 регуляризацией
    x = layers.Dense(64, activation = 'elu', kernel_regularizer = regularizers.l2(0.01))(x)
    # Dropout (убираем из процесса обучения неактивные/бесполезные нейроны, 50% от общего числа; регулируется первым аргументом функции)
    x = layers.Dropout(0.5)(x)
    # Создаем промежуточный выходной слой с обобщенными для дерева решений факторами и L2 регуляризацией
    features_output = layers.Dense(32, activation = 'elu', kernel_regularizer = regularizers.l2(0.001))(x)
    # Создаем выходной слой с вероятностью продолжения использования услуг компании
    output = layers.Dense(1, activation = 'sigmoid')(features_output)
    # Создаем модель нейросети с описанными выше слоями (выходные данные: вероятность)
    model = models.Model(inputs = inputs, outputs = output)
    model.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Создаем модель нейросети с описанными выше слоями (выходные данные: обобщенные факторы)
    feature_extractor = models.Model(inputs = inputs, outputs = features_output)
    # Возвращаем две модели нейросетей: вероястностей и обобщенных факторов
    return model, feature_extractor

def train_model(data):
    # Загружаем и предобрабатываем данные
    data = preprocess_data(data)
    features = data.drop(columns = ['client_id', 'target'])
    target = data['target']
    # Разделяем данные на обучающую и тестовую выборки (80% - обучающие данные, 20% - тестовые данные; регулируется переменной test_size)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.4)
    # Создаем модели нейросетей
    model, feature_extractor = build_neural_network(input_dim = X_train.shape[1])
    # Настраиваем параметр прерывания процесса обучения (если за 100 слоев коэффицицент ошибки не стабилизируется и не начнет уменьшаться, 
    #                                                    то прерываем обучение; регулируется переменной patience)
    early_stopping = callbacks.EarlyStopping(monitor = 'val_loss', patience = 100, restore_best_weights = True)
    # Обучаем модель нейросети вероятностей (с валидацией; количество эпох регулируется переменной epochs, количество единовременно анализируемых
    #                                        объектов – переменной batch_size)
    model.fit(X_train, y_train, epochs = 1000, batch_size = 1, validation_data = (X_test, y_test), callbacks = [early_stopping])
    # Извлекаем обобщенные факторы из соответствующей модели нейросети
    new_features_train = feature_extractor.predict(X_train)
    new_features_test = feature_extractor.predict(X_test)
    # Обучаем модель дерева на обобщенных факторах
    rf_model = RandomForestClassifier()
    rf_model.fit(new_features_train, y_train)
    # Прогнозируем бинарные значения с помощью модели дерева
    rf_predictions = rf_model.predict(new_features_test)
    # Прогнозируем вероятность с помощью соответствующей модели нейросети
    nn_predictions = model.predict(X_test).flatten()
    # Преобразуем прогнозы модели нейросети в бинарные значения
    nn_predictions_binary = (nn_predictions > 0.5).astype(int)
    # Делаем обобщенный прогноз как средний между прогнозами моделей дерева и нейросети
    combined_predictions = (nn_predictions_binary + rf_predictions) / 2 
    combined_predictions_binary = (combined_predictions > 0.5).astype(int)
    # Оцениваем точность обученного комплекса моделей
    accuracy = accuracy_score(y_test, combined_predictions_binary)
    print(f"Общая точность DNDT-системы: {accuracy * 100:.2f}%")
    # Возвращаем DNDT-модель (комплекс обученных моделей дерева и нейросети)
    return (model, feature_extractor, rf_model)

def main():
    # Указываем директорию с входными данными
    directory = 'training_input'
    # Загружаем входные данные
    data = load_all_data(directory)
    # Создаем и тренируем DNDT-модель
    model = train_model(data)
    # Выгружаем натренированную модель в директорию проекта
    dump(model, open('dndt_model.sav', 'wb'))

if __name__ == '__main__':
    main()