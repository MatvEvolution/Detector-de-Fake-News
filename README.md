
# Detecção de Fake News usando IA

Este projeto tem como objetivo testar, avaliar e comparar diversos modelos de IA treinados para detectar fake news usando vários modelos de aprendizado de máquina e aprendizado profundo. O objetivo é criar um modelo robusto que possa classificar com precisão os artigos de notícias como reais ou falsos. O projeto ainda se encontra em andamento.

## Índice

- [Introdução](#introdução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Modelos](#modelos)
- [Conjuntos de Dados](#conjuntos-de-dados)

## Introdução

Fake news se tornaram um problema significativo na era digital, espalhando desinformação e impactando a opinião pública. Este projeto utiliza inteligência artificial para identificar e classificar artigos de notícias falsos, ajudando os usuários a distinguir entre notícias reais e falsas.

## Estrutura do Projeto

```
main/
├── modelos/
│   ├── bert/
│   │   ├── treinamento/
│   │   └── teste/
│   ├── bilstm/
│   │   ├── treinamento/
│   │   └── teste/
│   ├── glove/
│   │   ├── treinamento/
│   │   └── teste/
│   ├── lstm stacked/
│   │   ├── treinamento/
│   │   └── teste/
│   ├── lstm vanilla/
│   │   └── treinamento/
│   ├── regressao logistica/
│   │   ├── treinamento/
│   │   └── teste/
│   └── word2vec/
├── pre-processamento/
|
├── site/
│   ├── cliente
│   ├── servidor
├── requirements.txt
├── README.md
└── LICENSE
```

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/MatvEvolution/Detector-de-Fake-News.git
    ```

2. Crie um ambiente virtual e ative-o:
    - Utilize por exemplo o anaconda.

3. Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```
4. Certifique-se de baixar o modelo do SpaCy para o idioma português:
    ```bash
    python -m spacy download pt_core_news_lg
    ```

## Modelos

O projeto inclui vários modelos para detecção de fake news:

BERT: Um modelo de linguagem baseado em transformers, conhecido por seu desempenho em várias tarefas de NLP.
BiLSTM (Word2Vec): Um modelo de LSTM bidirecional que captura dependências de longo alcance em ambas as direções.
BiLSTM (GloVe): Um modelo de embeddings pré-treinados combinado com uma arquitetura de rede neural.
LSTM Stacked: Uma arquitetura de LSTM com várias camadas empilhadas para capturar representações mais complexas.
LSTM Vanilla: Uma arquitetura básica de LSTM.
Regressão Logística: Um modelo simples e interpretável para classificação binária.

## Conjuntos de Dados

Usamos conjuntos de dados disponíveis publicamente para treinamento e avaliação, incluindo:

- [Fake.Br Corpus](https://github.com/roneysco/Fake.br-Corpus)
- [FakeRecogna](https://github.com/Gabriel-Lino-Garcia/FakeRecogna)
- [FakeTrueBR](https://github.com/jpchav98/FakeTrue.Br)

