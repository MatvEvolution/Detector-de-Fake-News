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
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

# Variáveis globais para armazenar os modelos e dados
model_bilstm = None
X_test_bilstm = None
y_test_bilstm = None
df_pre = None
df_csv = None
max_length = None
model_word2vec = None

def carrega_modelos():
    global model_bilstm, X_test_bilstm, y_test_bilstm, df_pre, df_csv, max_length, model_word2vec

    # Verifica se os modelos já foram carregados
    if model_bilstm is None or model_word2vec is None:
        # Encontra diretório atual
        atual_dir = os.getcwd()

        # Define caminhos para arquivos pickle e modelo
        caminho_pkl = os.path.join(atual_dir, "Pre-processamento\\df_pre_processado.pkl")
        caminho_csv = os.path.join(atual_dir, "Pre-processamento\\noticias_dados_limpos.csv")
        caminho_modelo = os.path.join(atual_dir, "Modelos\\BiLSTM\\Treinamento\\BiLSTM_model.h5")
        X_test_caminho = os.path.join(atual_dir, "Modelos\\BiLSTM\\Treinamento\\X_test_BiLSTM.npy")
        y_test_caminho = os.path.join(atual_dir, "Modelos\\BiLSTM\\Treinamento\\y_test_BiLSTM.npy")
        caminho_word2vec_model = os.path.join(atual_dir, "Pre-processamento\\model_word2vec.model")
        caminho_max_length = os.path.join(atual_dir, "Pre-processamento\\max_length.pkl")

        model_bilstm = load_model(caminho_modelo)
        X_test_bilstm = np.load(X_test_caminho)
        y_test_bilstm = np.load(y_test_caminho)

        with open(caminho_pkl, 'rb') as f:
            df_pre = pickle.load(f)

        df_csv = pd.read_csv(caminho_csv)

        with open(caminho_max_length, 'rb') as f:
            max_length = pickle.load(f)

        model_word2vec = Word2Vec.load(caminho_word2vec_model)
    
    return model_bilstm, X_test_bilstm, y_test_bilstm, df_pre, df_csv, max_length, model_word2vec

def execute_analysis(news):
    model_bilstm, X_test_bilstm, y_test_bilstm, df_pre, df_csv, max_length, model_word2vec = carrega_modelos()

    # Habilita suporte do tqdm para os métodos de progressão do pandas (como progress_apply)
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

        concatenated_df = pd.concat(chunks_processados) # Concatena os DataFrames processados

        df[coluna_texto] = concatenated_df[coluna_texto] # Atribui a coluna 'texto' processada de volta ao dataframe original
        
        # Remove tokens com espaços vazios
        print("Remover tokens com espaços vazios...")
        df[coluna_texto] = df[coluna_texto].progress_apply(lambda x: [token for token in x if token.strip()])

    # Cria um DataFrame 
    df_predict = pd.DataFrame(data={'Texto': [news]})

    # Faz o pré-processamento
    preprocess_data(df_predict, 'Texto')

    # Converte os dados para serem usados no modelo (rede neural)
    preprocessed_articles = df_predict['Texto'].tolist()

    # 'word_to_index' é um dicionário que mapeia cada palavra ao seu índice correspondente.
    word_to_index = {}

    # 'index_to_word' é um dicionário que mapeia cada índice à palavra correspondente.
    index_to_word = {}

    # Itera sobre a lista de palavras únicas obtida do modelo Word2Vec
    for i, word in enumerate(model_word2vec.wv.index_to_key):
        # Atribui a palavra ao índice i + 1 no dicionário 'word_to_index'.
        # Os índices começam em 1 para reservar o índice 0 para preenchimento (padding) quando necessário.
        word_to_index[word] = i + 1
        
        # Atribui o índice i + 1 à palavra no dicionário 'index_to_word'.
        index_to_word[i + 1] = word

    sequences_test = []
    for tokens in tqdm(preprocessed_articles):
        sequence = []
        for token in tokens:
            if token in word_to_index:
                sequence.append(word_to_index[token])
        sequences_test.append(sequence)

    padded_example = pad_sequences(sequences_test, maxlen=max_length, padding='post')

    # Faz a previsão usando o modelo
    predictions = model_bilstm.predict(padded_example)

    # Identifica a classe com a maior probabilidade
    predicted_class = np.argmax(predictions)   

    # Cria um objeto LabelEncoder
    le = LabelEncoder()

    # Transforma os labels para variáveis categóricas
    df_csv['label'] = le.fit_transform(df_csv['Categoria'])

    original_class = le.inverse_transform([predicted_class]) 
    result = original_class[0]
        
    result_prediction_final = predictions[0][predicted_class] * 100

    related_news = [{"title": "Título da Notícia Relacionada", "summary": "Resumo da Notícia Relacionada", "image": "url_da_imagem"}]

    return result, result_prediction_final, related_news
