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
    
    if st.button("Начать предсказание"):
        newss = extract_keywords(news)
        class_pred = ['Bad', 'Good'][classifier.predict(newss)]
        answer_class = st.empty()
        answer_class.text(class_pred)

        if st.button("Правильно"):
            update_model(news, class_pred, 1)
        if st.button("Неправильно"):
            update_model(news, class_pred, -1)
        if st.button("Не знаю"):
            update_model(news, 'net', 0)
    
def update_model(news, predicted_class, feedback):
    try:
        data = pd.read_excel('dataset.xlsx')
    except FileNotFoundError:
        data = pd.DataFrame(columns=['news', 'predicted_class'])

    if predicted_class == 'Bad':
        label = -1
    elif predicted_class == 'Good':
        label = 1
    else:
        label = 0

    label *= feedback

    new_entry = pd.DataFrame({'news': [news], 'predicted_class': [predicted_class]})
    updated_data = pd.concat([data, new_entry], ignore_index=True)

    updated_data.to_excel('dataset.xlsx', index=False, mode='a', header=not os.path.exists('dataset.xlsx'))


if __name__ == "__main__":
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny2',
        tokenizer_path='cointegrated/rubert-tiny2',
        n_classes=2,
        epochs=4,
        model_save_path='bert.pt'
    )
    predict_news(classifier)
