import streamlit as st
import pandas as pd
import joblib

# Configuração da Página
st.set_page_config(page_title="Predição Imobiliária", page_icon="🏡", layout="wide")

st.title("🏡 Simulador de Preços Imobiliários")
st.write("Bem-vindo ao dashboard interativo! Preencha as características do imóvel na barra lateral para ver o preço sugerido pela Inteligência Artificial.")

# 1. Carregando os Modelos Salvos
@st.cache_resource
def load_models():
    return joblib.load('modelos_imobiliarios.pkl')

try:
    modelos = load_models()
except FileNotFoundError:
    st.error("Arquivo de modelo não encontrado! Certifique-se de exportar os modelos no Notebook primeiro.")
    st.stop()

# 2. Criando o Painel Lateral (Sidebar) para os inputs
st.sidebar.header("📋 Características do Imóvel")

area = st.sidebar.number_input("Área (SqFt)", min_value=100.0, max_value=20000.0, value=1500.0, step=50.0)
quartos = st.sidebar.slider("Quartos", min_value=1, max_value=10, value=3)
ano = st.sidebar.number_input("Ano de Construção", min_value=1900, max_value=2026, value=2015)
mobilia = st.sidebar.selectbox("Nível de Mobília", options=[0, 1, 2], format_func=lambda x: ["Sem Mobília", "Semi-Mobiliado", "Mobiliado"][x])
piscina = st.sidebar.radio("Tem Piscina?", options=[0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

st.sidebar.markdown("---")
st.sidebar.subheader("Códigos de Região")
tipo_loc = st.sidebar.number_input("Tipo de Localidade (Código)", min_value=0, max_value=10, value=1)
locais = st.sidebar.number_input("Bairro/Cidade (Código)", min_value=0, max_value=2000, value=2)
tipos = st.sidebar.number_input("Tipo do Imóvel (Código)", min_value=0, max_value=10, value=0)

# 3. Preparando os dados para o modelo
dados_nova_casa = {
    'Area': [area],
    'Quartos': [quartos],
    'Ano_Construcao': [ano],
    'Furnishing_Numerico': [mobilia],
    'tipo_localidade': [tipo_loc],
    'locais': [locais],
    'tipos': [tipos],
    'Piscina': [piscina]
}
nova_casa_df = pd.DataFrame(dados_nova_casa)

st.markdown("### 📊 Resultado da Simulação")

# 4. Escolhendo qual modelo usar
st.markdown("Escolha o modelo de Inteligência Artificial para fazer a previsão:")
escolha_modelo = st.selectbox("", options=["Todos (Fazer uma média)"] + list(modelos.keys()))

if st.button("🚀 Calcular Preço do Imóvel"):
    with st.spinner("Calculando o preço justo..."):
        
        if escolha_modelo == "Todos (Fazer uma média)":
            preco_lmr = modelos["RLM"].predict(nova_casa_df)[0]
            preco_rf = modelos["RF"].predict(nova_casa_df)[0]
            preco_gb = modelos["GB"].predict(nova_casa_df)[0]
            
            media = (preco_lmr + preco_rf + preco_gb) / 3
            
            st.success(f"**Valor de Mercado Sugerido:** R$ {media:,.2f}")
            
            # Mostrando a opinião de cada um
            col1, col2, col3 = st.columns(3)
            col1.metric("Regressão Múltipla", f"R$ {preco_lmr:,.2f}")
            col2.metric("Random Forest", f"R$ {preco_rf:,.2f}")
            col3.metric("Gradient Boosting", f"R$ {preco_gb:,.2f}")
            
        else:
            modelo_escolhido = modelos[escolha_modelo]
            preco = modelo_escolhido.predict(nova_casa_df)[0]
            st.success(f"**Valor de Mercado Sugerido:** R$ {preco:,.2f}")
            st.info(f"Modelo utilizado: {escolha_modelo}")
