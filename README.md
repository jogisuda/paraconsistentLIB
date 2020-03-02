# paraconsistentLIB

Esta é uma biblioteca criada para o python 3, com o intuito de trazer uma nova ferramenta de engenharia de características baseada no uso de Lógica Paraconsistente. Para entender melhor esta nova técnica, ler [1].

Funções disponíveis:
    -ParaconsistentAnalysis(number_of_classes, number_of_feature_vectors_in_class, dimension_of_each_feature_vector, c): retorna a distância do conjunto providenciado (vetor) de características ao ponto (1, 0) no plano Paraconsistente. Se verbose = True, a função imprimirá valores de alpha e beta também.
    -BestParaconsistent(): ..., analisa a função ParaconsistentAnalysis() para todos os possíveis subconjuntos de características de tamanho 1, 2,..., n, onde n é o tamanho do conjunto original de características. Se verbose = True, a função imprimirá valores de alpha e beta para cada subconjunto analisado.
    

Exemplos de uso:
  -checar ex1.py


Referências:

[1] - Guido, R.C; "Paraconsistent feature engineering", IEEE Signal Process Mag., vol. 36, no. 1, pp. 154-158, 2019.
