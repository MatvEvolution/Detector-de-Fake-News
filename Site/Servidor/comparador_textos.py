from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import nltk
import os
nltk.download('punkt')

# Função para pré-processamento de texto
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return tokens

# Carregar modelo de embedding de palavras pré-treinado
# Encontra diretório atual
atual_dir = os.getcwd()

caminho = os.path.join(atual_dir, "Modelos\\Word2Vec\\modelo_word2vec.model")

model = Word2Vec.load(caminho)

def calcular_similaridade_com_lista(texto_referencia, lista_de_dicionarios):
    # Inicializar uma lista para armazenar os resultados de similaridade
    resultados_similaridade = []


    for dicionario in lista_de_dicionarios:
        # Acessar o texto diretamente do dicionário
        texto = dicionario.get("texto", "")  # Retorna uma string vazia se a chave 'texto' não estiver presente
        
        if texto_referencia.strip() and texto.strip():  # Verifica se ambos os textos não estão vazios
            similarity = model.wv.n_similarity(preprocess_text(texto_referencia), preprocess_text(texto))
        else:
            similarity = 0  # Define a similaridade como 0 se um dos textos estiver vazio
        
        # Adicionar a informação de similaridade ao dicionário
        dicionario['similaridade'] = similarity

        # Adicionar o dicionário atual à lista de resultados
        resultados_similaridade.append(dicionario)

    # Ordenar a lista de resultados com base na similaridade (maior similaridade primeiro)
    resultados_similaridade = sorted(resultados_similaridade, key=lambda x: x['similaridade'], reverse=True)

    return resultados_similaridade  # Retornar a lista de dicionários ordenados com a similaridade

