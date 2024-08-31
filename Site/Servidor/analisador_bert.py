import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Função para carregar o modelo BERT
def load_bert_model(model_save_path):
    # Inicializa um modelo BERT para classificação
    model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased')
    
    # Carrega os pesos do modelo salvo
    model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
    model.eval()  # Coloca o modelo em modo de avaliação
    return model

# Função para pré-processar e tokenizar o texto
def preprocess_and_tokenize(texto, tokenizer, max_length=128):
    texto = texto.lower()
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=max_length)
    return inputs

# Função para fazer a previsão
def predict(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax().item()
    prediction_confidence = torch.nn.functional.softmax(logits, dim=1)[0, predicted_class_idx].item()
    return predicted_class_idx, prediction_confidence

# Função principal para executar a análise
def execute_analysis_bert(news):
    # Diretório atual e caminho para o CSV
    atual_dir = os.getcwd()
    caminho_csv = os.path.join(atual_dir, "Pre-processamento\\noticias_dados_limpos.csv")
    
    # Carrega dataframe
    df = pd.read_csv(caminho_csv)
    
    # Caminho do modelo
    model_save_path = os.path.join(atual_dir, "Modelos\\BERT\\Treinamento\\bert_model.bin")
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    
    # Carrega o modelo
    model = load_bert_model(model_save_path)
    
    # Pré-processa e tokeniza o texto
    inputs = preprocess_and_tokenize(news, tokenizer)
    
    # Faz a previsão
    predicted_class_idx, prediction_confidence = predict(inputs, model)
    
    # Mapeia o índice da classe prevista para a categoria original
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Categoria'])
    le.classes_ = le.classes_  # Manter classes como no treino
    original_class = le.inverse_transform([predicted_class_idx])[0]
    
    # Resultados finais
    result = original_class
    result_prediction_final = prediction_confidence * 100
    
    related_news = [{"title": "Título da Notícia Relacionada", "summary": "Resumo da Notícia Relacionada", "image": "url_da_imagem"}]
    
    return result, result_prediction_final, related_news
