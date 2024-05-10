"""Colocar o nome dos elementos do grupo"""

import random
import math
import matplotlib.pyplot as plt

#valor exemplificativo para a velocidade de aprendizagem (tambem lhe podíamos ter chamado new)
alpha = 0.2

#------------------CÓDIGO GENÉRICO PARA CRIAR, TREINAR E USAR UMA REDE COM UMA CAMADA ESCONDIDA------------
"""Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
tal como foi discutido na teorica, as saidas destas estas unidades estao sempre a -1.
Por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
def make(nx, nz, ny):
    #a rede neuronal é um dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    return nn

#Funcao de activacao (sigmoide)
def sig(inp):
    return 1.0/(1.0 + math.exp(-inp))

"""Função que recebe uma rede nn e um padrao de entrada inp (uma lista) 
e faz a propagacao da informacao para a frente ate as saidas"""
def forward(nn, inp):
    #copia a informacao do vector de entrada in para a listavector de inputs da rede nn  
    nn['x']=inp.copy()
    nn['x'].append(-1)
    
    #calcula a activacao da unidades escondidas
    nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1)
    
    #calcula a activacao da unidades de saida
    nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
"""Funcao que recebe uma rede nn com as activacoes calculadas e a lista output de saidas pretendidas e calcula os erros
na camada escondida e na camada de saida"""
def error(nn, output):
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
"""Funcao que recebe uma rede com as activacoes e erros calculados e actualiza as listas de pesos"""
def update(nn):
    nn['wzx'] = [[w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    
"""Funcao que realiza uma iteracao de treino para um dado padrao de entrada inp com saida desejada output"""
def iterate(i, nn, inp, output):
    forward(nn, inp)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, inp, output, nn['y']))

#-----------------------CÓDIGO QUE PERMITE CRIAR E TREINAR REDES PARA APRENDER AS FUNÇÕES BOOLENAS--------------------
"""Funcao que cria uma rede 2x2x1 e treina a função lógica AND
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_and(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
"""Funcao que cria uma rede 2x2x1 e treina um OR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_or(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

"""Funcao que cria uma rede 2x2x1 e treina um XOR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_xor(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net
    
    
#-------------------------CÓDIGO QUE IRÁ PERMITIR CRIAR UMA REDE PARA APRENDER A CLASSIFICAR COGUMELOS---------  

""" Dicionário que sumaria a informação sobre os atributos. Cada chave correponde à posição do um atributo no exemplo
e o valor correspondente é um outro dicionário em que as chaves são os valores possíveis do atributo
e os valores são um inteiro que pode ser usada para facilitar a codificação binária do atributo """

dicionario = {0 : {'b':0, 'c': 1, 'x':2, 'f': 3, 'k': 4, 's':5}, #cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
                  1: {'f':0, 'g': 1, 'y':2, 's': 3}, #cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
                  #cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
                  2: {'n':0, 'b': 1, 'c':2, 'g': 3, 'r': 4, 'p':5, 'u':6, 'e':7, 'w':8, 'y':9},    
                  3: {'t':0, 'f': 1}, #bruises: bruises=t,no=f   
                  #odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
                  4: {'a':0, 'l': 1, 'c':2, 'y': 3, 'f': 4, 'm':5, 'n':6, 'p':7, 's':8}, 
                  #gill-attachment: attached=a,descending=d,free=f,notched=n
                  5: {'a':0, 'd': 1, 'f':2, 'n': 3}, 
                  #gill-spacing: close=c,crowded=w,distant=d
                  6: {'c':0, 'w': 1, 'd':2}, 
                  #gill-size: broad=b,narrow=n
                  7: {'b':0, 'n': 1}, 
                  #gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
                  8: {'k':0, 'n': 1, 'b':2, 'h': 3, 'g': 4, 'r':5, 'o':6, 'p':7, 'u':8, 'e':9, 'w': 10, 'y':11}, 
                  #stalk-shape: enlarging=e,tapering=t
                  9: {'e':0, 't': 1}, 
                  #stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
                  10: {'b':0, 'c': 1, 'u':2, 'e': 3, 'z': 4, 'r':5, '?':6}, 
                  #stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
                  11: {'f':0, 'y': 1, 'k':2, 's': 3},
                  #stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
                  12: {'f':0, 'y': 1, 'k':2, 's': 3}, 
                  #stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
                  13: {'n':0, 'b': 1, 'c':2, 'g': 3, 'o': 4, 'p':5, 'e':6, 'w':7, 'y':8},
                  #stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
                  14: {'n':0, 'b': 1, 'c':2, 'g': 3, 'o': 4, 'p':5, 'e':6, 'w':7, 'y':8},
                  #veil-type: partial=p,universal=u
                  15: {'p':0, 'u': 1}, 
                  #veil-color: brown=n,orange=o,white=w,yellow=y
                  16: {'n':0, 'o': 1, 'w':2, 'y': 3},
                  #ring-number: none=n,one=o,two=t
                  17: {'n':0, 'o': 1, 't': 1}, 
                  #ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
                  18: {'c':0, 'e': 1, 'f':2, 'l': 3, 'n': 4, 'p':5, 's':6, 'z':7},
                  #spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
                  19: {'k':0, 'n': 1, 'b':2, 'h': 3, 'r': 4, 'o':5, 'u':6, 'w':7, 'y':8},
                  #population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
                  20: {'a':0, 'c': 1, 'n':2, 's': 3, 'v': 4, 'y':5},
                  #habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
                  21: {'g':0, 'l': 1, 'm':2, 'p': 3, 'u': 4, 'w':5, 'd':6}}


"""Funcao principal do nosso programa para classificar cogumelos: cria os conjuntos de treino e teste, chama
a funcao que cria e treina a rede e, por fim, a funcao que a testa.
A funcao recebe como argumento o ficheiro correspondente ao dataset que deve ser usado, os tamanhos das camadas de entrada, escondida e saída,
o numero de epocas que deve ser considerado no treino, os tamanhos do conjunto de treino e teste e 
o intervalo de iterações """
def run_mushrooms(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size, print_step):
    pass
    

"""Funcao que cria os conjuntos de treino e de de teste a partir dos dados
armazenados em f (mushrooms.csv). A funcao le cada linha, tranforma-a numa lista de valores e 
chama a funcao translate para a colocar no formato adequado para o padrao de treino. 
Estes padroes são colocados numa lista.
A função recebe como argumentos o nº de exemplos que devem ser considerados no conjunto de treino --->x e
o nº de exemplos que devem ser considerados no conjunto de teste ------> y
Finalmente, devolve duas listas, uma com x padroes (conjunto de treino)
e a segunda com y padrões (conjunto de teste). Atenção que x+y não pode ultrapassar o nº de cogumelos 
disponível no dataset"""
def build_sets(f,x,y):
    pass

"""A função translate recebe cada lista de valores simbólicos transforma-a num padrão de treino. 
Cada padrão é uma lista com o seguinte formato [padrao_de_entrada, classe_do_cogumelo, padrao_de_saida]
O enunciado do trabalho explica de que forma deve ser obtido o padrão de entrada
"""
def translate(lista):
    pass

"""Cria a rede e chama a funçao iterate para a treinar. A função recebe como argumento os conjuntos de treino e teste,
os tamanhos das camadas de entrada, escondida e saída e o número de épocas que irão ser usadas para fazer o treino"""
def train_mushrooms(input_size, hidden_size, output_size, training_set, test_set, epochs):
    pass

"""Recebe o padrao de saida da rede e devolve a classe com que a rede classificou o cogumelo.
Devolve a classe que corresponde ao indice da saida com maior valor."""
def retranslate(out):
    pass

"""Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste ou treino.
Para cada padrao do conjunto chama a funcao forward e determina a classe do cogumelo
que corresponde ao maior valor da lista de saida. A classe determinada pela rede deve ser comparada com a classe real,
sendo contabilizado o número de respostas corretas. A função calcula a percentagem de respostas corretas"""    
def test_mushrooms(net, test_set, printing = True):
    pass

if __name__ == "__main__":
    #Vamos treinar durante 1000 épocas uma rede para aprender a função logica AND
    #Faz testes para números de épocas diferentes e para as restantes funções lógicas já implementadas
    rede_AND = train_and(1000)
    #Agora vamos ver se ela aprendeu bem
    tabela_verdade = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
    for linha in tabela_verdade:
        forward(rede_AND, list(linha))
        print('A rede determinou %s para a entrada %d AND %d quando devia ser %d'%(rede_AND['y'], linha[0], linha[1], tabela_verdade[linha]))