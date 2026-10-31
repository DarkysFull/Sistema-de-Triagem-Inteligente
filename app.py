import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import folium
from streamlit_folium import st_folium

# Configuração da Página
st.set_page_config(page_title="Ocorrências - Bairro Nova Cidade", page_icon="🏙️", layout="centered")

# Inicializando o contador de ocorrências na sessão do Streamlit
if 'contador_ocorrencias' not in st.session_state:
    st.session_state.contador_ocorrencias = 1

st.title("🏙️ Central de Ocorrências Urbanas")
st.markdown("### Bairro Nova Cidade - Matão / SP")
st.write("Registre problemas de infraestrutura, saneamento ou arborização com foto e geolocalização.")

# Base de treinamento expandida e robusta para corrigir a classificação da IA
textos_exemplo = [
    # Infraestrutura / Vias
    "buraco enorme na rua atrapalhando os carros", 
    "asfalto quebrado e esburacado na via pública",
    "cratera aberta no meio da rua causando risco de acidente",
    "buraco na rua com risco de acidente",
    "piso cedendo e afundamento de asfalto",
    
    # Saneamento / Água
    "falta de água na torneira há três dias", 
    "torneira seca sem pingo de água em casa",
    "água da rua saindo turva, suja e com cheiro forte",
    "vazamento de esgoto na calçada a céu aberto",
    "cano estourado jorrando água na rua",
    
    # Arborização
    "árvore com galhos caindo na fiação elétrica",
    "galhos grandes de árvore atrapalhando a passagem",
    "necessidade de poda de árvore na calçada",
    "raiz de árvore levantando o piso da calçada",
    
    # Iluminação Pública
    "lâmpada da praça queimada e tudo muito escuro",
    "poste de luz apagado na rua inteira à noite",
    "fiação exposta no poste soltando faísca",
    "luminária pública quebrada gerando insegurança"
]

classes_exemplo = [
    # Infraestrutura / Vias
    "Infraestrutura / Vias", 
    "Infraestrutura / Vias", 
    "Infraestrutura / Vias", 
    "Infraestrutura / Vias", 
    "Infraestrutura / Vias",
    
    # Saneamento / Água
    "Saneamento / Água", 
    "Saneamento / Água", 
    "Saneamento / Água", 
    "Saneamento / Água", 
    "Saneamento / Água",
    
    # Arborização
    "Arborização", 
    "Arborização", 
    "Arborização", 
    "Arborização",
    
    # Iluminação Pública
    "Iluminação Pública", 
    "Iluminação Pública", 
    "Iluminação Pública", 
    "Iluminação Pública"
]

vetorizador = TfidfVectorizer()
X_treino = vetorizador.fit_transform(textos_exemplo)
modelo_ia = MultinomialNB()
modelo_ia.fit(X_treino, classes_exemplo)

# Formulário de Registro
with st.form("form_ocorrencia"):
    st.markdown("#### 1. Identificação do Cidadão")
    email_cidadao = st.text_input("Seu e-mail (para receber cópia e acompanhar a ocorrência):", placeholder="seu_email@outlook.com")
    
    st.markdown("#### 2. Dados do Problema")
    categoria_manual = st.selectbox(
        "Categoria Principal:", 
        ["Infraestrutura / Vias (ex: Buracos)", "Saneamento / Água (ex: Qualidade/Falta)", "Arborização", "Iluminação Pública", "Outros"]
    )
    
    localizacao = st.text_input("Endereço / Ponto de Referência:", placeholder="Ex: Rua Principal, próximo ao nº 120 - Nova Cidade")
    
    descricao = st.text_area(
        "Descrição detalhada da ocorrência:", 
        placeholder="Descreva o que está acontecendo (ex: buraco grande abrindo no asfalto ou água saindo turva)..."
    )
    
    st.markdown("#### 3. Registro Fotográfico")
    foto_arquivo = st.file_uploader("Envie uma foto do problema (ou tire pelo celular):", type=["jpg", "jpeg", "png"])
    
    st.markdown("#### 4. Indique a Localização no Mapa (Matão - SP)")
    st.write("Clique no mapa para marcar o local aproximado do incidente:")
    
    m = folium.Map(location=[-21.6033, -48.3644], zoom_start=15)
    m.add_child(folium.LatLngPopup())
    
    mapa_dados = st_folium(m, height=300, width=700)
    
    submitted = st.form_submit_button("Registrar e Enviar Ocorrência")

# Ação ao submeter o formulário
if submitted:
    if not localizacao or not descricao or not email_cidadao:
        st.error("Por favor, preencha o seu e-mail, o endereço e a descrição do problema antes de enviar.")
    else:
        # Geração do Número de Protocolo Formatado (ex: #0000001)
        numero_protocolo = f"#{st.session_state.contador_ocorrencias:07d}"
        
        # Processamento com IA (NLP)
        texto_vetorizado = vetorizador.transform([descricao])
        categoria_prevista = modelo_ia.predict(texto_vetorizado)[0]
        
        # Avaliação de nível de urgência baseado em palavras-chave
        if any(palavra in descricao.lower() for palavra in ["urgente", "perigo", "enorme", "total", "risco", "vazamento", "suja"]):
            urgencia = "Crítica"
        elif any(palavra in descricao.lower() for palavra in ["pequeno", "leve"]):
            urgencia = "Baixa"
        else:
            urgencia = "Média"

        st.success(f"Ocorrência registrada com sucesso! Protocolo: **{numero_protocolo}**")
        
        # Exibição dos resultados da Triagem
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Categoria Identificada", value=categoria_prevista)
        with col2:
            st.metric(label="Nível de Urgência (IA)", value=urgencia)
            
        st.info(f"📋 **Protocolo:** {numero_protocolo}\n\n📍 **Endereço:** {localizacao}\n\n📝 **Relato:** {descricao}")
        
        # Exibição da foto enviada (se houver)
        if foto_arquivo is not None:
            st.markdown("#### 📷 Foto Anexada:")
            st.image(foto_arquivo, caption="Evidência do problema", use_column_width=True)

        # Captura de coordenadas do mapa se o usuário clicou
        lat, lng = None, None
        if mapa_dados and mapa_dados.get("last_clicked"):
            lat = mapa_dados["last_clicked"]["lat"]
            lng = mapa_dados["last_clicked"]["lng"]
            st.write(f"🗺️ **Coordenadas GPS:** Latitude: {lat:.6f}, Longitude: {lng:.6f}")

        # Configuração de Envio de E-mail via Outlook (SMTP)
        try:
            remetente = "vitor_hugo4@outlook.com" 
            destinatario_principal = "vitor_hugo@outlook.com" 
            
            msg = MIMEMultipart()
            msg['From'] = remetente
            msg['To'] = destinatario_principal
            msg['Cc'] = email_cidadao  
            msg['Subject'] = f"[Ocorrência Nova Cidade] - Protocolo {numero_protocolo} - Urgência: {urgencia}"
            
            corpo = f"""
            Olá! Uma nova ocorrência foi registrada no Bairro Nova Cidade (Matão-SP).
            
            Protocolo: {numero_protocolo}
            Categoria: {categoria_prevista}
            Nível de Urgência: {urgencia}
            Endereço: {localizacao}
            Relato do Morador: {descricao}
            E-mail de Contato do Morador: {email_cidadao}
            Coordenadas GPS: {f"Lat: {lat}, Lng: {lng}" if lat else "Não informada no mapa"}
            
            Atenciosamente,
            Central de Ocorrências Urbanas - Nova Cidade
            """
            
            msg.attach(MIMEText(corpo, 'plain'))
            destinatarios = [destinatario_principal, email_cidadao]
            
            # Insira aqui a sua senha do Outlook (ou senha de aplicativo caso utilize autenticação em duas etapas)
            senha_app = "B@zingado123" 
            
            # Conexão SMTP utilizando os servidores oficiais do Outlook
            servidor = smtplib.SMTP('smtp-mail.outlook.com', 587, timeout=10)
            servidor.starttls()
            servidor.login(remetente, senha_app)
            servidor.sendmail(remetente, destinatarios, msg.as_string())
            servidor.quit()
            
            st.success(f"E-mail enviado com sucesso! Protocolo {numero_protocolo} enviado para {destinatario_principal} com cópia para {email_cidadao}.")
            
            # Incrementa o contador para a próxima ocorrência
            st.session_state.contador_ocorrencias += 1
            
        except Exception as e:
            st.warning(f"Ocorrência registada sob o protocolo **{numero_protocolo}** com sucesso!)
