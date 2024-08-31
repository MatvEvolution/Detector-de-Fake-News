from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import os
import spacy

# Carrega o modelo spaCy para português
nlp = spacy.load('pt_core_news_lg')

# Função para pré-processamento de texto
def preprocess_text(text):
    doc = nlp(text.lower())
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

# Carrega modelo de embedding de palavras pré-treinado
# Encontra diretório atual
atual_dir = os.getcwd()

caminho = os.path.join(atual_dir, "pre-processamento\\model_word2vec.model")

model = Word2Vec.load(caminho)

def calcular_similaridade_com_lista(texto_referencia, lista_de_dicionarios):
    # Inicializa uma lista para armazenar os resultados de similaridade
    resultados_similaridade = []


    for dicionario in lista_de_dicionarios:
        # Acessa o texto diretamente do dicionário
        texto = dicionario.get("texto", "")  # Retorna uma string vazia se a chave 'texto' não estiver presente
        
        if texto_referencia.strip() and texto.strip():  # Verifica se ambos os textos não estão vazios
            similarity = model.wv.n_similarity(preprocess_text(texto_referencia), preprocess_text(texto))
        else:
            similarity = 0  # Define a similaridade como 0 se um dos textos estiver vazio
        
        # Adiciona a informação de similaridade ao dicionário
        dicionario['similaridade'] = similarity

        # Adiciona o dicionário atual à lista de resultados
        resultados_similaridade.append(dicionario)

    # Ordena a lista de resultados com base na similaridade (maior similaridade primeiro)
    resultados_similaridade = sorted(resultados_similaridade, key=lambda x: x['similaridade'], reverse=True)

    return resultados_similaridade  # Retorna a lista de dicionários ordenados com a similaridade

