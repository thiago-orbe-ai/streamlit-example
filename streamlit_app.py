# IMPORTAR AS BIBLIOTECAS NECESSÁRIAS E O ALGORIMO K-NN
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# CRIAR FUNÇÕES NECESSÁRIAS PARA CARREGAR DADOS E TREINAR MODELO.

# Função para carregar o dataset
def carregar_dados():
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
  return dataframe
  

# Função para treinar o modelo
def treinar_modelo():
  df = carregar_dados()
  X_atributos_preditores = df.iloc[:,1:4].values
  y_atributo_alvo = df.iloc[:,4].values
  # Normalização Min-Max para a os preditores
  scaler = sklearn.preprocessing.MinMaxScaler().fit(X_atributos_preditores)
  X_atributos_preditores_scaled = scaler.transform(X_train)
  modelo_knn_classificacao = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
  modelo_knn_classificacao.fit(X_atributos_preditores_scaled,y_atributo_alvo)
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
# Aplicar a normalização Min-Max dos preditores nos novos dados
new_data = [salario, idade, valor_emprestimo]
new_data_scaled = scaler.transform(new_data)


# Botão para realizar a avaliação de crédito.
botao_realizar_avaliacao = st.button("Realizar avaliação")

# SE o botão seja acionado.
# 01.Coletar todos os dados que o usuário informou.
# 02.Usar os dados para predizer o resultado. Crédito aprovado ou reprovado.
# 03.Mostrar o resultado da avaliação.
if botao_realizar_avaliacao:
    resultado = modelo.predict([new_data_scaled])
    st.subheader("Resultado: ")
    if resultado == 0:
      resultado_avaliacao = "crédito aprovado"
    else:
      resultado_avaliacao = "crédito reprovado"
      
    st.write(resultado_avaliacao)
