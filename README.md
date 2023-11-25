
Jogo da Velha - Classificação usando Multilayer Perceptron (MLP)
Este repositório contém um projeto de classificação utilizando um Multilayer Perceptron (MLP) para prever o desfecho de configurações no jogo da velha. O conjunto de dados é composto por 958 amostras representando todas as possíveis configurações do tabuleiro.

Estrutura do Projeto
Conjunto de Dados:

Os dados foram carregados a partir do arquivo CSV tic-tac-toe.csv.
Um mapeamento foi definido para converter os valores ("o", "b", "x", "negativo", "positivo") para números inteiros.
Treinamento do Modelo MLP:

O conjunto de dados foi dividido em recursos (X) e rótulos (y).
Utilizou-se o MLPClassifier do scikit-learn para criar e treinar o modelo.
A acurácia do modelo foi avaliada usando 20% dos dados como conjunto de teste.
Avaliação do Modelo:

Métricas de avaliação, incluindo acurácia e um relatório de classificação, foram exibidas.
Teste com Dados Arbitrários:

O modelo permite que o usuário insira dados arbitrários para classificação.
O resultado indica se os dados constituem uma vitória para "x" ou não.
Execução do Código
Para executar o código, siga as instruções apresentadas durante a execução, inserindo os dados de acordo com as orientações fornecidas. O código continuará a executar até que a opção "sair" seja selecionada.

Requisitos
Certifique-se de ter as seguintes bibliotecas instaladas:

pip install pandas scikit-learn

Referências
UCI Machine Learning Repository

Documentação:

[Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
[scikit-learn](https://scikit-learn.org/stable/)
