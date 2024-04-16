import pandas as pd
import os
import string
import spacy
import unicodedata
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.tokenize import word_tokenize

# Variáveis globais
word_to_index = None
index_to_word = None
sequences = None
model = None
max_length = None

def load_word2vec_models(caminho1, caminho2, caminho_sequences):
    # Carregar dicionários de arquivos usando pickle
    global word_to_index
    global index_to_word
    global sequences

    with open(caminho1, 'rb') as handle:
        word_to_index = pickle.load(handle)

    with open(caminho2, 'rb') as handle:
        index_to_word = pickle.load(handle)

    with open(caminho_sequences, 'rb') as handle:
        sequences = pickle.load(handle)

    return word_to_index, index_to_word, sequences

def load_bi_lstm_model(caminho_modelo, X_test_caminho, y_test_caminho):
    # Carrega modelo Bi-LSTM
    global model
    global max_length

    model = load_model(caminho_modelo)
    X_test = np.load(X_test_caminho)
    y_test = np.load(y_test_caminho)

    # Calcula uma porcentagem de 95% dos comprimentos das sequências (preserva 95% das sequências)
    max_length = int(np.percentile([len(seq) for seq in sequences], 95))

    return model, X_test, y_test


def execute_analysis(news):
    global word_to_index, index_to_word, sequences, model, max_length

    # Encontra diretório atual
    atual_dir = os.getcwd()

    # Define caminhos para arquivos pickle e modelo
    caminho_pkl = os.path.join(atual_dir, "Pre-processamento\\noticias_pre_processadas_df.pkl")
    caminho_csv = os.path.join(atual_dir, "Pre-processamento\\noticias_dados_limpos.csv")
    caminho1 = os.path.join(atual_dir, "Modelos\\Word2Vec\\word_to_index.pickle")
    caminho2 = os.path.join(atual_dir, "Modelos\\Word2Vec\\index_to_word.pickle")
    caminho_sequences = os.path.join(atual_dir, "Modelos\\Word2Vec\\sequences.pickle")
    caminho_modelo = os.path.join(atual_dir, "Modelos\\BiLSTM\\Treinamento\\modelo_BiLSTM.h5")
    X_test_caminho = os.path.join(atual_dir, "Modelos\\BiLSTM\\Treinamento\\X_test_BiLSTM.npy")
    y_test_caminho = os.path.join(atual_dir, "Modelos\\BiLSTM\\Treinamento\\y_test_BiLSTM.npy")

    if word_to_index is None or index_to_word is None or sequences is None:
        word_to_index, index_to_word, sequences = load_word2vec_models(caminho1, caminho2, caminho_sequences)

    if model is None or max_length is None:
        model, X_test, y_test = load_bi_lstm_model(caminho_modelo, X_test_caminho, y_test_caminho)

    # Carregar dataframe salvo em formato pickle
    df = pd.read_pickle(caminho_pkl)

    # Habilita suporte do tqdm para os métodos de progressão do pandas (como progress_aplly)
    tqdm.pandas()

    # Carrega modelo de linguagem 'pt_core_news_lg' do spacy para processamento de texto em português
    # Desabilita os componentes 'parser' e 'ner', já que não são necessários para a lematização
    modelo_spacy_nlp = spacy.load("pt_core_news_lg", disable=["parser", "ner"])

    def preprocess_data(df, coluna_texto):
        """
        Realiza o pré-processamento dos dados de um texto em um Dataframe do Pandas.
        Remove pontuação, números e palavras comuns (stop words), converte para minúsculas, remove 
        acentos e símbolos diversos, e aplica lematização.
        """
        # Remove pontuação
        print("Removendo pontuação...")
        traducao = str.maketrans('', '', string.punctuation)
        df[coluna_texto] = df[coluna_texto].progress_apply(lambda x: x.translate(traducao))

        # Remove números
        print("Removendo números...")
        traducao = str.maketrans('', '', string.digits)
        df[coluna_texto] = df[coluna_texto].progress_apply(lambda x: x.translate(traducao))

        # Remove acentos e símbolos diversos
        print("Removendo acentos e símbolos diversos...")
        def remove_acentos_e_simbolos(text):
            try:
                # Normaliza a string para a forma NFKD e mantém apenas caracteres que não são diacríticos
                # nem combinam caracteres com diacríticos
                return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c) and unicodedata.category(c) != 'Mn')
            except TypeError:
                # Se não for possível normalizar um caractere, retorna uma string vazia
                return ''
        df[coluna_texto] = df[coluna_texto].progress_apply(lambda x: remove_acentos_e_simbolos(x))
        
        # Converte para minúsculas
        print("Convertendo para minúsculas...")
        df[coluna_texto] = df[coluna_texto].progress_apply(lambda x: x.lower())

        # Lematização
        print("Computando Lematização...")
        def lematizar_texto(doc):
            return [token.lemma_ for token in doc if not token.is_stop]

        n_chunks = 10  # Ajuste esse valor de acordo com o tamanho da base de dados e a memória disponível no sistema
        chunks = np.array_split(df, n_chunks) # Divide o dataframe em várias partes

        chunks_processados = []
        for i, chunk in enumerate(chunks):
            print(f"Processando segmento {i + 1} de {n_chunks}")
            chunk_processado = chunk.copy() # Cria uma cópia para realizar o processamento
            
            # Aplica a função 'lematizar_texto' a cada documento processado pelo spaCy (usando 'spacy_nlp_model.pipe') e
            # atribui os resultados (uma lista de palavras lematizadas) à coluna 'coluna_texto' do DataFrame 'chunks_processados'.
            # O tqdm é utilizado para exibir uma barra de progresso durante o processamento dos documentos.
            chunk_processado[coluna_texto] = [lematizar_texto(doc) for doc in tqdm(modelo_spacy_nlp.pipe(chunk[coluna_texto].astype(str), batch_size=100, disable=['parser', 'ner']), total=len(chunk[coluna_texto]))]

            # Junta as partes em uma lista, para formar o dataframe final
            chunks_processados.append(chunk_processado)

        concatenated_df = pd.concat(chunks_processados) # Concatenar os DataFrames processados

        df[coluna_texto] = concatenated_df[coluna_texto] # Atribuir a coluna 'texto' processada de volta ao dataframe original
        
        # Remover tokens com espaços vazios
        print("Remover tokens com espaços vazios...")
        df[coluna_texto] = df[coluna_texto].progress_apply(lambda x: [token for token in x if token.strip()])

    # Crie um DataFrame com uma linha e a coluna 'data'
    df_predict = pd.DataFrame(data={'Texto': [news]})

    # Faz pré-processamento
    preprocess_data(df_predict, 'Texto')

    # Conversão dos dados para serem usados no modelo (rede neural)
    preprocessed_articles = df_predict['Texto'].tolist()

    sequences_test = []
    for tokens in tqdm(preprocessed_articles):
        sequence = []
        for token in tokens:
            if token in word_to_index:
                sequence.append(word_to_index[token])
        sequences_test.append(sequence)

    padded_example = pad_sequences(sequences_test, maxlen=max_length, padding='post')

    # Fazer a previsão usando o modelo
    predictions = model.predict(padded_example)

    # Identificar a classe com a maior probabilidade
    predicted_class = np.argmax(predictions)   

    # Carregar dataframe salvo em formato csv
    df = pd.read_csv(caminho_csv)

    from sklearn.preprocessing import LabelEncoder

    # Cria um objeto LabelEncoder
    le = LabelEncoder()

    # Transforma os labels para variáveis categóricas
    df['label'] = le.fit_transform(df['Categoria'])

    original_class = le.inverse_transform([predicted_class]) 
    result = original_class[0]
        
    result_prediction_final = predictions[0][predicted_class] * 100


    # Aqui você poderia fazer outras operações necessárias, como coletar notícias relacionadas
    related_news = [{"title": "Título da Notícia Relacionada", "summary": "Resumo da Notícia Relacionada", "image": "url_da_imagem"}]

    return result, result_prediction_final, related_news
