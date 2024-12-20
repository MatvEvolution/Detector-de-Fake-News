from flask import Flask, request, jsonify, render_template
import os
import analisador
import analisador_bert
from palavras_chave import extrair_palavras_chave
from coleta_noticias_relacionadas import coletar_noticias_paralelo
from comparador_textos import calcular_similaridade_com_lista
from flask import redirect, url_for


# Obtém o caminho do diretório atual do arquivo Python Flask
current_dir = os.path.dirname(os.path.abspath(__file__))

# Obtém o diretório pai do diretório atual
parent_dir = os.path.split(current_dir)[0]

# Define o caminho relativo para o diretório de templates
template_dir = os.path.join(parent_dir, 'Cliente')

# Define o caminho relativo para o diretório de templates
static_dir = os.path.join(template_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)


# Variável global para armazenar o progresso da análise
progresso_analise = "Aguardando análise..."

@app.route('/')
def index():
    return render_template('detector_fakenews.html')

# Função para análise de notícias falsas
def analyze_news(news):
    import subprocess
    import sys

    global progresso_analise

    def update_progress(progress):
        global progresso_analise
        progresso_analise = progress


    update_progress("Coletando Palavras-chave")
    palavras_chave = extrair_palavras_chave(news)
    print(palavras_chave)


    update_progress("Coletando Notícias Relacionadas")
    resultados_noticias = coletar_noticias_paralelo(palavras_chave)
    print(resultados_noticias)

    update_progress("Verificando Semelhança com as Noticias Coletadas")
    # Reordena a lista de dicionários com base na similaridade
    lista_ordenada = calcular_similaridade_com_lista(news, resultados_noticias)

    # Exibe os textos reordenados
    for dicionario in lista_ordenada:
        print("Texto:", dicionario['texto'])
        print("Similaridade:", dicionario['similaridade'])
        print()

    
    update_progress("Realizando Análise com o modelo de Inteligência Artificial")
    # Caminho para o arquivo Flask externo
    ##flask_file_path = "Site\\Servidor\\analisador.py" #caminho para o modelo bilstm

    flask_file_path = "Site\\Servidor\\analisador_bert.py"#caminho para o modelo bert

    # Executa o arquivo Flask como um processo separado
    process = subprocess.Popen([sys.executable, flask_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Aguarda até que o processo termine e obtenha a saída
    output, error = process.communicate()

    # Verifica se houve algum erro
    if error:
        return "Erro ao executar Análise: " + str(error)
    else:
        #resultados = analisador.execute_analysis(news) #versao bilstm
        resultados = analisador_bert.execute_analysis_bert(news) #versao bert
        result = resultados[0]
        result_prediction_final = resultados[1]
        progresso_analise = "Concluído"
        result_prediction_final = round(result_prediction_final, 2)
        return lista_ordenada, result, result_prediction_final


# Rota para receber a notícia e enviar o resultado da análise
@app.route('/analyze', methods=['POST'])
def analyze():
    news = request.form['news']
    lista_ordenada, result, result_prediction_final = analyze_news(news)

    if lista_ordenada is None:
        lista_ordenada = []
    
    return render_template('results.html', result=result, result_prediction_final=result_prediction_final, lista_ordenada=lista_ordenada)

# Rota para fornecer o progresso atual
@app.route('/progress', methods=['GET'])
def get_progress():
    global progresso_analise
    return jsonify({"progressText": progresso_analise})

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    app.run(debug=True)
