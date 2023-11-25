import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# Carregando os dados do arquivo CSV
dados = pd.read_csv('tic-tac-toe.csv')

# Definir o mapeamento
mapeamento = {'o': -1, 'b': 0, 'x': 1, 'negativo': -1, 'positivo': 1}

# Substituir os valores de acordo com o mapeamento
new_dados = dados.replace(mapeamento)

# print(dados)
# print(new_dados)

# Separar os dados em recursos (X) e rótulos (y)
x = new_dados.iloc[:, :-1]  # Recursos (todas as colunas, exceto a última)
y = new_dados.iloc[:, -1]   # Rótulos (última coluna)

# print(x)
# print(y)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(x_treino, y_treino)

# Fazer previsões nos dados de teste
previsoes = mlp.predict(x_teste)

print("Métrica de avaliação para verificação da acurácia do Multilayer Percepton (scikit-learn):\n")

# Calcular a acurácia
acuracia = accuracy_score(y_teste, previsoes)
print("Acurácia:", acuracia * 100, "%", "\n")

# Exibir outras métricas, se necessário
relatorio = classification_report(y_teste, previsoes)
print("Relatório de Classificação:\n", relatorio)

print("Testando o modelo com dados arbitrários:\n")

while True:

    # Permitir ao usuário inserir dados arbitrários para classificação
    print("Insira os dados a serem classificados (x, o, b) separados por vírgula: (Exemplo: x,x,o,o,b,b,x,x,b)")

    entrada = input().strip().split(',')

    print()

    # Converter a entrada do usuário em um array de números com base no mapeamento
    input_numerico = [mapeamento[val] for val in entrada]

    # Realizar a classificação com base no modelo treinado
    resultado = mlp.predict([input_numerico])

    # Imprimir o resultado
    if resultado[0] == 1:
        print("Com base no modelo, os dados inseridos constituem uma vitória de x (sim).\n")
    else:
        print("Com base no modelo, os dados inseridos não constituem uma vitória de x (não).\n")

    print("Caso deseja sair do codigo, favor digitar 'sair'. Para continuar, favor apertar 'enter'!")
    saida = input()

    if saida == "sair":
        break
    else: continue
