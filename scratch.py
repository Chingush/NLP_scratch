import pandas as pd
from Class_Bert import BertClassifier
from sklearn.model_selection import train_test_split
import streamlit as st


def predict_news():
    # Загрузка модели
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny2',
        tokenizer_path='cointegrated/rubert-tiny2',
        n_classes=2,
        epochs=4,
        model_save_path='bert.pt'
    )

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
    st.write("Это текст, который должен отображаться на странице")

def update_model(new_data, label, answer):
    # Загрузка данных для начального обучения
    data = pd.read_excel('dataset.xlsx')
    if(label == 'Bad'):
       label = -1
    elif(label == 'Good'):
       label = 1
    else:
        label = 0
    label = label * answer
    new_entry = pd.DataFrame({'news': [new_data], 'class': [label]})
    updated_data = pd.concat([data, new_entry], ignore_index=True)
    print('Не вышло, не вышло!')

    updated_data.to_excel('dataset.xlsx', index=False)
    
    #X = updated_data['news']
    #y = updated_data['class']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #classifier.preparation(X_train, y_train, X_test, y_test)
    #classifier.fit()
    #classifier.train()

    if __name__ == "__main__":
        predict_news()

