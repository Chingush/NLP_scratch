import pandas as pd
from Class_Bert import BertClassifier
from sklearn.model_selection import train_test_split
import streamlit as st


def predict_news(classifier):
    
    if st.button("Переобучить модель"):
        st.write("Модель переобучается...")
        retrain_model(classifier)
    # Вывод формы для ввода новости
    news = st.text_area("Вставьте новость", height=100)

    # Обработка пользовательского ввода
    if st.button("Начать предсказание"):
        class_pred = ["Bad", "Good"][classifier.predict(news)]
        answer_class = st.empty()
        answer_class.text(class_pred)

        correct_button = st.button("Правильно")
        incorrect_button = st.button("Неправильно")
        net_button = st.button("Не знаю")

        if correct_button:
            update_model(news, class_pred, 1)
        elif incorrect_button:
            update_model(news, class_pred, -1)
        elif net_button:
            update_model(news, 'net', 0)
    
    # Добавляем вывод для отладки

def retrain_model(classifier):
    # Загрузка данных для обучения
    data = pd.read_excel('dataset.xlsx')
    X = data['news']
    y = data['class']

    # Разделение данных на обучающий и тестовый наборы (если это необходимо)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание экземпляра модели и обучение её на данных

    
    classifier.retrain_model(X_train, y_train, X_valid, y_valid)


    st.write("Модель переобучена.")


def update_model(new_data, label, answer):
    # Загрузка данных из файла
    try:
        data = pd.read_excel('dataset.xlsx')
    except FileNotFoundError:
        # Если файл отсутствует, создаем пустой DataFrame
        data = pd.DataFrame(columns=['news', 'class'])

    if(label == 'Bad'):
        label = -1
    elif(label == 'Good'):
        label = 1
    else:
        label = 0
    label = label * answer

    # Создание новой записи и добавление её к существующим данным
    new_entry = pd.DataFrame({'news': [new_data], 'class': [label]})
    updated_data = pd.concat([data, new_entry], ignore_index=True)

    # Сохранение обновленных данных в файл
    updated_data.to_excel('dataset.xlsx', index=False)



if __name__ == "__main__":
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny2',
        tokenizer_path='cointegrated/rubert-tiny2',
        n_classes=2,
        epochs=4,
        model_save_path='bert.pt'
    )
    predict_news(classifier)
