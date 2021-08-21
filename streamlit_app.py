# IMPORTAR AS BIBLIOTECAS NECESSÁRIAS E O ALGORIMO K-NN
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# CRIAR FUNÇÕES NECESSÁRIAS PARA CARREGAR DADOS E TREINAR MODELO.

# Função para carregar o dataset
@st.cache
def carregar_dados():
  return pd.read_csv('/content/dados_de_credito_limpo.csv')

# Função para treinar o modelo
def treinar_modelo():
  df = carregar_dados()
  df=(df-df.min())/(df.max()-df.min())
  X_atributos_preditores = df.iloc[:,1:4].values
  y_atributo_alvo = df.iloc[:,4].values
  modelo_knn_classificacao = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
  modelo_knn_classificacao.fit(X_atributos_preditores,y_atributo_alvo)
  return modelo_knn_classificacao

# MODELO DE CLASSIFICAÇÃO
# treinando o modelo
modelo = treinar_modelo()
  
# SITE
# Título do site
st.title("Site para classificar empréstimo.")

# Subtítulo 
st.subheader("Insira seus dados.")

# Recebendo os dados do usuário.
salario = st.number_input("Salário", value=0)
idade = st.number_input("Idade", value=0)
valor_emprestimo = st.number_input("Valor empréstimo", value=0)

# Botão para realizar a avaliação de crédito.
botao_realizar_avaliacao = st.button("Realizar avaliação")

# SE o botão seja acionado.
# 01.Coletar todos os dados que o usuário informou.
# 02.Usar os dados para predizer o resultado. Crédito aprovado ou reprovado.
# 03.Mostrar o resultado da avaliação.
if botao_realizar_avaliacao:
    resultado = modelo.predict([[salario,idade,valor_emprestimo]])
    st.subheader("Resultado: ")
    if resultado == 0:
      resultado_avaliacao = "crédito aprovado"
    else:
      resultado_avaliacao = "crédito reprovado"
      
    st.write(resultado_avaliacao)
    
    
# 01.Execute o site a partir de seu arquivo `site_analise_de_credito.py`.
!streamlit run site_analise_de_credito.py &>/dev/null&

# 02.Crie o link para poder acessar o site criado.
!npx localtunnel --port 8501
