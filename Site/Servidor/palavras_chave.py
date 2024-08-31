import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from bs4 import BeautifulSoup
import os

nltk.download('stopwords')
nltk.download('punkt')

def extrair_palavras_chave(texto, quantidade_palavras_chave=10):
    # Remove pontuações e outros caracteres indesejados
    tokenizer = RegexpTokenizer(r'\w+')
    palavras = tokenizer.tokenize(texto)

    # Remove stopwords (palavras comuns que geralmente não são úteis para identificar o conteúdo)
    stopwords_pt = set(stopwords.words('portuguese'))
    palavras_sem_stopwords = [palavra.lower() for palavra in palavras if palavra.lower() not in stopwords_pt]

    # Contagem das palavras
    contagem_palavras = Counter(palavras_sem_stopwords)

    # Obtenção das palavras-chave mais comuns
    palavras_chave = contagem_palavras.most_common(quantidade_palavras_chave)
    
    return palavras_chave

# Carrega o arquivo HTML
# Obtém o diretório pai do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.split(current_dir)[0]

# Define o caminho relativo para o diretório de templates
cliente_dir = os.path.join(parent_dir, 'Cliente//detector_fakenews.html')
with open(cliente_dir, 'r', encoding='utf-8') as arquivo_html:
    conteudo_html = arquivo_html.read()

# Analisa o HTML
soup = BeautifulSoup(conteudo_html, 'html.parser')

# Encontra o campo de texto no HTML (supondo que esteja dentro de uma tag <textarea>)
campo_texto = soup.find('textarea').text

# Chamada da função para extrair palavras-chave
palavras_chave = extrair_palavras_chave(campo_texto)

# Exibição das palavras-chave
print("Principais palavras-chave:")
for palavra, frequencia in palavras_chave:
    print(f"{palavra}: {frequencia}")
