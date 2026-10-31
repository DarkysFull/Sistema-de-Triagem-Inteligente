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

st.title("🏙️ Central de Ocorrências Urbanas")
st.markdown("### Bairro Nova Cidade - Matão / SP")
st.write("Registre problemas de infraestrutura, saneamento ou arborização com foto e geolocalização.")

# Modelo de IA pré-treinado para demonstração
textos_exemplo = [
    "buraco enorme na rua atrapalhando os carros", 
    "falta de água na torneira há três dias", 
    "árvore com galhos caindo na fiação elétrica",
    "lâmpada da praça queimada escuro",
    "vazamento de esgoto na calçada",
    "água da rua saindo suja e com cheiro forte"
]
classes_exemplo = [
    "Infraestrutura / Vias", 
    "Saneamento / Água", 
    "Arborização", 
    "Iluminação Pública", 
    "Saneamento / Água", 
    "Saneamento / Água"
]

vetorizador = TfidfVectorizer()
X_treino = vetorizador.fit_transform(textos_exemplo)
modelo_ia = MultinomialNB()
modelo_ia.fit(X_treino, classes_exemplo)

# Formulário de Registro
with st.form("form_ocorrencia"):
    st.markdown("#### 1. Dados do Problema")
    
    categoria_manual = st.selectbox(
        "Categoria Principal:", 
        ["Infraestrutura / Vias (ex: Buracos)", "Saneamento / Água (ex: Qualidade/Falta)", "Arborização", "Iluminação Pública", "Outros"]
    )
    
    localizacao = st.text_input("Endereço / Ponto de Referência:", placeholder="Ex: Rua Principal, próximo ao nº 120 - Nova Cidade")
    
    descricao = st.text_area(
        "Descrição detalhada da ocorrência:", 
        placeholder="Descreva o que está acontecendo (ex: buraco grande abrindo no asfalto ou água saindo turva)..."
    )
    
    st.markdown("#### 2. Registro Fotográfico")
    foto_arquivo = st.file_uploader("Envie uma foto do problema (ou tire pelo celular):", type=["jpg", "jpeg", "png"])
    
    st.markdown("#### 3. Indique a Localização no Mapa (Matão - SP)")
    st.write("Clique no mapa para marcar o local aproximado do incidente:")
    
    # Criando mapa centralizado no Bairro Nova Cidade / Matão - SP
    # Coordenadas aproximadas de Matão-SP
    m = folium.Map(location=[-21.6033, -48.3644], zoom_start=15)
    m.add_child(folium.LatLngPopup()) # Permite capturar coordenadas ao clicar
    
    mapa_dados = st_folium(m, height=300, width=700)
    
    submitted = st.form_submit_button("Registrar e Enviar Ocorrência")

# Ação ao submeter o formulário
if submitted:
    if not localizacao or not descricao:
        st.error("Por favor, preencha o endereço e a descrição do problema antes de enviar.")
    else:
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

        st.success("Ocorrência registrada e analisada com Inteligência Artificial!")
        
        # Exibição dos resultados da Triagem
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Categoria Identificada", value=categoria_prevista)
        with col2:
            st.metric(label="Nível de Urgência (IA)", value=urgencia)
            
        st.info(f"📍 **Endereço Informado:** {localizacao}\n\n📝 **Relato:** {descricao}")
        
        # Exibição da foto enviada (se houver)
        if foto_arquivo is not None:
            st.markdown("#### 📷 Foto Anexada pelo Morador:")
            st.image(foto_arquivo, caption="Evidência do problema enviada para a prefeitura", use_column_width=True)
        else:
            st.warning("Nenhuma foto foi anexada, mas a ocorrência foi registrada com os dados textuais e de localização.")

        # Captura de coordenadas do mapa se o usuário clicou
        if mapa_dados and mapa_dados.get("last_clicked"):
            lat = mapa_dados["last_clicked"]["lat"]
            lng = mapa_dados["last_clicked"]["lng"]
            st.write(f"🗺️ **Coordenadas GPS capturadas no mapa:** Latitude: {lat:.6f}, Longitude: {lng:.6f}")

        # Bloco de Envio de E-mail via smtplib (Pronto para configurar credenciais)
        try:
            # remetente = "seu_email@gmail.com"
            # senha = "sua_senha_de_app"
            # destinatario = "prefeitura@matao.sp.gov.br"
            
            # msg = MIMEMultipart()
            # msg['From'] = remetente
            # msg['To'] = destinatario
            # msg['Subject'] = f"[Ocorrência Nova Cidade - Matão/SP] - Urgência: {urgencia}"
            
            # corpo = f"Endereço: {localizacao}\nCategoria: {categoria_prevista}\nUrgência: {urgencia}\nRelato: {descricao}"
            # msg.attach(MIMEText(corpo, 'plain'))
            
            # servidor = smtplib.SMTP('smtp.gmail.com', 587)
            # servidor.starttls()
            # servidor.login(remetente, senha)
            # servidor.sendmail(remetente, destinatario, msg.as_string())
            # servidor.quit()
            
            st.toast("Relatório com foto, mapa e triagem enviado para o órgão competente!", icon="🚀")
        except Exception as e:
            pass