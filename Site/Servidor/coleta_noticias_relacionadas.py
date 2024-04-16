import requests
from bs4 import BeautifulSoup
import concurrent.futures

def coletar_noticias_g1(palavras_chave, lista_noticias):
     # G1
    url_base_g1 = 'https://g1.globo.com/'

    busca = ' '.join(palavra for palavra, _ in palavras_chave)

    response = requests.get(url_base_g1 + 'busca/?q=' + busca + '&order=recent&species=notícias')

    if response.status_code == 200:
        content = response.content
        site = BeautifulSoup(content, 'html.parser')

        # HTML da noticia
        noticias = site.findAll('div', attrs={'class': 'widget--info__text-container'})

        for i, noticia in enumerate(noticias):
            if i >= 3:  # Limita a coleta a 3 notícias
                break

            # Titulo
            titulo = noticia.find('div', attrs={'class': 'widget--info__title product-color'})
            titulo_texto = titulo.text.strip() if titulo else ''

            # Link original
            link_original = noticia.find('a')['href'] if noticia.find('a') else ''

            # Link redirecionado após o clique
            link_redirecionado, texto = obter_link_redirecionado("https:" + link_original, url_base_g1)

            # Subtitulo
            subtitulo = noticia.find('p', attrs={'class': 'widget--info__description'})
            subtitulo_texto = subtitulo.text.strip() if subtitulo else ''

            # Adiciona a notícia à lista
            lista_noticias.append({'titulo': titulo_texto, 'subtitulo': subtitulo_texto, 'link': link_redirecionado, 'texto': texto})

    return lista_noticias

def coletar_noticias_sbt(palavras_chave, lista_noticias):
    # SBT News
    url_base_sbt = 'https://sbtnews.sbt.com.br/'

    busca = ' '.join(palavra for palavra, _ in palavras_chave)

    response = requests.get(url_base_sbt + 'search?q=' + busca)

    if response.status_code == 200:
        content = response.content
        site = BeautifulSoup(content, 'html.parser')

        # Encontra a div principal que contém todas as notícias
        div_noticias_principais = site.find('div', class_='LatestNews_lastestNewsPageItems__jCdvo')

        if div_noticias_principais:
            # Encontra todas as divs dentro da div principal que representam notícias individuais
            divs_noticias = div_noticias_principais.find_all('div', class_=None)

            for i, div_noticia in enumerate(divs_noticias):
                # Verifica se atingiu o limite de 3 notícias
                if i >= 3:
                    break

                # Titulo
                titulo = div_noticia.find('div', attrs={'class': 'if-title-container'})
                titulo_texto = titulo.text.strip() if titulo else ''

                # Link original
                link_original = div_noticia.find('a')['href'] if div_noticia.find('a') and div_noticia.find('a')['href'].startswith('/noticia/') else ''

                # Link redirecionado após o clique
                link_redirecionado = url_base_sbt + link_original

                # Subtitulo
                subtitulo = div_noticia.find('span', attrs={'class': 'if-subtitle'})
                subtitulo_texto = subtitulo.text.strip() if subtitulo else ''

                
                link_redirecionado, texto = extrair_texto(url_base_sbt, link_redirecionado)

                # Adiciona a notícia à lista
                lista_noticias.append({'titulo': titulo_texto, 'subtitulo': subtitulo_texto, 'link': link_redirecionado, 'texto': texto})


    return lista_noticias

def coletar_noticias_cnn(palavras_chave, lista_noticias):
    # CNN Brasil
    url_base_cnn = 'https://www.cnnbrasil.com.br/'

    busca = ' '.join(palavra for palavra, _ in palavras_chave)

    response = requests.get(url_base_cnn + '?s=' + busca + '&orderby=date&order=desc')

    if response.status_code == 200:
        content = response.content
        site = BeautifulSoup(content, 'html.parser')

        noticias = site.find_all('li', attrs={'class': 'home__list__item'})

        for i, noticia in enumerate(noticias):
            if i >= 3:
                break

            # Titulo
            titulo = noticia.find('h3', attrs={'class': 'news-item-header__title market__new__title'})
            titulo_texto = titulo.text.strip() if titulo else ''

            # Link original
            link_original = noticia.find('a', attrs={'class': 'home__list__tag'})['href'] 

            link_redirecionado, texto = extrair_texto(url_base_cnn, link_original)

            # Subtitulo
            indice_quebra = texto.find('\n')
            subtitulo_texto = texto[:indice_quebra]
            texto = texto[indice_quebra+1:]

            # Adiciona a notícia à lista
            lista_noticias.append({'titulo': titulo_texto, 'subtitulo': subtitulo_texto, 'link': link_redirecionado, 'texto': texto})


    return lista_noticias

def coletar_noticias_paralelo(palavras_chave):
    lista_noticias = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(coletar_noticias_g1, palavras_chave, lista_noticias)
        executor.submit(coletar_noticias_sbt, palavras_chave, lista_noticias)
        executor.submit(coletar_noticias_cnn, palavras_chave, lista_noticias)

    return lista_noticias

from selenium import webdriver

def configure_chrome_options():

    CaminhoWD = 'C:\\Users\\mathe\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe'

    CaminhoBin = 'C:\\Users\\mathe\\Downloads\\chrome-win64\\chrome-win64\\chrome.exe'
    options = webdriver.ChromeOptions()
    options.binary_location = CaminhoBin
    options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path=CaminhoWD, options= options)

    return driver

def obter_link_redirecionado(link_original, url_base):
    try:
       
        link_redirecionado, text = extrair_texto(url_base, link_original)

        return link_redirecionado, text
        
    except Exception as e:
        print(f"Erro ao obter link redirecionado para '{link_original}': {e}")
        return link_original, ''


def extrair_texto(url_base, link_redirecionado):
    text = ''  # Inicializa a variável text com um valor padrão
    if(url_base == 'https://g1.globo.com/'):
        # Extrair texto
        try:
           
            driver = configure_chrome_options()
        
            driver.get(link_redirecionado)
            content = driver.page_source
            site = BeautifulSoup(content, 'html.parser')

            link_redirecionado = driver.current_url

            noticias = site.find_all('p', attrs={'class': 'content-text__container'})
            text = '\n'.join([noticia.text for noticia in noticias])

            driver.quit()
        except:
            pass
    elif(url_base == 'https://sbtnews.sbt.com.br/'):
        # Extrair texto
        try:
            
            driver = configure_chrome_options()
        
            driver.get(link_redirecionado)
        
            content = driver.page_source
            site = BeautifulSoup(content, 'html.parser')

            articles = site.find_all('article')
            for article in articles:
                # Encontra todas as tags <div> dentro do <article>
                divs = article.find_all('div', attrs= {'class': 'blocks-renderer-container'})
                
                # Extrai o texto de cada <p> e faz algo com ele
                text = '\n'.join([div.text for div in divs])
            driver.quit()
        except:
            pass
    elif(url_base == 'https://www.cnnbrasil.com.br/'):
        # Extrair texto
        try:
            
            driver = configure_chrome_options()
        
            driver.get(link_redirecionado)
        
            content = driver.page_source
            site = BeautifulSoup(content, 'html.parser')

            noticias = site.find('article')
            noticia_texto = noticias.find_all('p')
            text = '\n'.join([noticia.text for noticia in noticia_texto])

            driver.quit()
        except:
            pass
    return link_redirecionado, text