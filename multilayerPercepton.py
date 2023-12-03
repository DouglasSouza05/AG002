import pandas as pd
import numpy as np
import warnings  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore", category=UserWarning)

dados = pd.read_csv('tic-tac-toe.csv')

# Definir o mapeamento
mapeamento = {"o": -1, "b": 0, "x": 1, "negativo": -1, "positivo": 1}

# Substituir os valores de acordo com o mapeamento
new_dados = dados.replace(mapeamento)

# Separar os dados em recursos (X) e rótulos (y)
x = new_dados.iloc[:, :-1]  # Recursos (todas as colunas, exceto a última)
y = new_dados.iloc[:, -1]   # Rótulos (última coluna)

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

def matriz(entrada):

    matriz = np.array(entrada).reshape(3, 3)

    for linha in matriz:
        print("+---+---+---+")
        print("| {} | {} | {} |".format(linha[0], linha[1], linha[2]))
    print("+---+---+---+")

while True:

    # Permitir ao usuário inserir dados arbitrários para classificação
    print("Insira os dados a serem classificados (x, o, b) separados por vírgula: (Exemplo: x,x,o,o,b,b,x,x,b)")

    entrada = input().strip().split(',')

    # Verifica se cada elemento em "entrada" corresponde a uma chave em "mapeamento". Iterável de valores booleanos. A função all() retorna True somente se todos os elementos do iterável foram True.
    if all(val in mapeamento for val in entrada):
        # Converter a entrada do usuário em um array de números com base no mapeamento
        input_numerico = [mapeamento[val] for val in entrada]

        # Realizar a classificação com base no modelo treinado
        resultado = mlp.predict([input_numerico])
    else:
        print()
        print("Entrada Inválida!!! Insira somente 'x', 'o' ou 'b'. Atentar-se a forma correta! \n")
        continue

    print()

    if resultado[0] == 1:
        print("Com base no modelo, os dados inseridos constituem uma vitória de x (sim). \n")
    elif resultado[0] == -1:
        print("Com base no modelo, os dados inseridos não constituem uma vitória de x (não). \n")

    print("Matriz Resultante: \n")
    matriz(entrada)
    print()

    print("Caso deseja sair do codigo, favor digitar 'Sair'. Para continuar, favor apertar tecla 'Enter'!")
    saida = input()

    if saida.lower() == "sair":
        print()
        print("Você escolheu Sair. Até mais!")
        break
    else: 
        print("Você escolheu Continuar...")
        continue