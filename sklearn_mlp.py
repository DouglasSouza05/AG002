import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregando os dados do arquivo CSV
dados = pd.read_csv('tic-tac-toe.csv')

# Convertendo 'x' em 1, 'o' em -1 e 'b' em 0
le = LabelEncoder()
dados_encoded = dados.apply(le.fit_transform)

# Separando features e rótulos
X = dados_encoded.iloc[:, :-1]
y = dados_encoded['resultado']

# Definindo os nomes das colunas
nomes_colunas = ['posicao1', 'posicao2', 'posicao3', 'posicao4', 'posicao5', 'posicao6', 'posicao7', 'posicao8', 'posicao9']

# Atribuindo os nomes das colunas aos dados
X.columns = nomes_colunas

# Dividindo os dados em conjuntos de treinamento e teste (80% treinamento, 20% teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo MLP
modelo = MLPClassifier(hidden_layer_sizes=(4, 3, 1), activation='logistic', max_iter=1000, random_state=42)
modelo.fit(X_treino, y_treino)

# Função para permitir que o usuário insira dados de forma arbitrária
def obter_dados_do_usuario():
    entrada_usuario = input("Digite as posições no tabuleiro separadas por vírgula (por exemplo, 'x,o,x,b,b,o,b,x,x'): ")
    entrada_usuario = entrada_usuario.split(',')
    entrada_usuario = [1 if x == 'x' else -1 if x == 'o' else 0 for x in entrada_usuario]
    return [entrada_usuario]

# Realizando a previsão com os dados inseridos pelo usuário
dados_usuario = obter_dados_do_usuario()
previsao = modelo.predict(dados_usuario)

# Imprimindo o resultado da previsão
if previsao[0] == 1:
    print("Sim, houve vitória do X.")
else:
    print("Não, não houve vitória do X.")
