// Abre e fecha janela Flutuante de Laoding
function openModal() {
    var modal = document.getElementById("modal");
    modal.style.display = "block";

    // Exibir progresso na janela flutuante
    var progressDiv = document.getElementById("progress");
}

function closeModal() {
    var modal = document.getElementById("modal");
    modal.style.display = "none";
}


// Função para enviar a notícia e iniciar a atualização de progresso
function submitNews() {
    var news = document.getElementById("newsInput").value;
    
    // Envia notícia para o servidor para análise
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: 'news=' + encodeURIComponent(news)
    })
    .then(response => response.json())
    .then(data => {
        // Mostra resultado da análise
        document.getElementById("result").innerText = data.result;

        // Se a análise for concluída, redireciona para a página de resultados
        if (data.progressText === "Concluído") {
            window.location.href = '/results'; 
        }
    })
    .catch(error => console.error('Error:', error));

    // Inicia atualização do progresso periodicamente
    updateProgress();
}

// Função que atualiza o progresso
function updateProgress() {
    // Busca progresso no servidor
    fetch('/progress', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        // Atualiza texto do progresso
        document.getElementById("progress").innerText = "Progresso: " + data.progressText;

        // Se a análise ainda não está concluída, continua atualizando o progresso
        if (data.progressText !== "Concluído") {
            setTimeout(updateProgress, 4000); 
        }
    })
    .catch(error => console.error('Error fetching progress:', error));
}

// Chama submitNews() quando o formulário é enviado
document.addEventListener("DOMContentLoaded", function() {
    var newsForm = document.getElementById("newsForm");
    if (newsForm) {
        newsForm.addEventListener("submit", function(event) {
            submitNews();
        });
    } else {
        console.error("Elemento #newsForm não encontrado.");
    }

    // Abrir a janela flutuante quando a página carrega
    var modal = document.getElementById('warningModal');
    modal.style.display = 'block';

    // Fechar a janela flutuante ao clicar no botão de fechar
    var closeBtn = document.querySelector('.close');
    if (closeBtn) {
        closeBtn.onclick = function() {
            closeWarningModal(); // Chamando a função closeModal() aqui
        }
    } else {
        console.error("Elemento .close não encontrado.");
    }
});

// Função para fechar a janela flutuante
function closeWarningModal() {
    var modal = document.getElementById("warningModal");
    if (modal) {
        modal.style.display = 'none';
    }
}

