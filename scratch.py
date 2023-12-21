import os
import pandas as pd
from Class_Bert import BertClassifier
import streamlit as st
import requests


def extract_keywords(text):
    response = requests.post("https://wrapapi.com/use/nu_dayte_pochitaty/extract_keywords/extract/0.0.7", json={
        "input_parameters": text,
        "wrapAPIKey": "QCMnwFgcGaOfk6ytNVehETNKuEZ8n40h"
    })

    if response.status_code == 200:
        response_data = response.json()
        if 'rawData' in response_data:
            raw_data = response_data['rawData']
            if 'responses' in raw_data:
                responses = raw_data['responses']
                keyword_list = []
                for item in responses:
                    if 'body' in item:
                        keyword_list.append(item['body'])
                # Объединение элементов списка в одну строку через запятую
                texxt = ', '.join(keyword_list)
                return texxt

    return texxt  

def predict_news(classifier):
    news = st.text_area("Вставьте новость", height=100)
    keywords = extract_keywords(news)

    if st.button("Начать предсказание"):
        predicted_class = ["Bad", "Good"][classifier.predict(keywords)]
        answer_class = st.empty()
        answer_class.text(predicted_class)

        feedback_options = ["Правильно", "Неправильно", "Не знаю"]
        feedback = st.radio("Фидбек:", feedback_options)

        if feedback == "Правильно":
            update_model(news, predicted_class, 1)
        elif feedback == "Неправильно":
            update_model(news, predicted_class, -1)
        elif feedback == "Не знаю":
            update_model(news, 'Неизвестно', 0)

def update_model(news, predicted_class, feedback):
    try:
        data = pd.read_excel('dataset.xlsx')
    except FileNotFoundError:
        data = pd.DataFrame(columns=['news', 'class'])

    if predicted_class == 'Bad':
        label = -1
    elif predicted_class == 'Good':
        label = 1
    else:
        label = 0

    label *= feedback

    new_entry = pd.DataFrame({'news': [news], 'label': [label]})
    updated_data = pd.concat([data, new_entry], ignore_index=True)

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
