import random
import matplotlib.pyplot as plt

pop_size = 200            # Tamanho da populacao, deve ser par
n_gen = 2000                # Numero de geracoes

pc = 0.75                 # Probabilidade de recombinacao em percentagem
pm = 0.0001                 # Probabilidade de mutacao em percentagem


def run_ga(prob_file):
    """Função que deve ser executada para resolver o problema.
    Executa o ciclo principal do GA durante n_gen gerações chamando as outras funções implementadas.
    
    Args:
        prob_file (str): Caminho do arquivo contendo informações sobre o problema.
    """
    
    # prob é um dicionário que guarda os dados relativos ao problema a resolver
    # 'n' corresponde ao número de itens a transportar
    # 'l' é o limite de peso da mochila
    # 'pesos' é uma lista com os pesos dos diversos itens
    # 'valores' é uma lista com os valores ganhos por transportar os itens
    prob = {'n': 0, 'l': 0, 'pesos': [], 'valores': []}
    
   
    make_prob(prob_file, prob)
    
    # Imprime os dados formatados em colunas
    print("-" * (20 + 10 + 15))
    print("| {:<15} | {:<10} | {:<10} |".format("Item", "Peso(Kg)", "Lucro(€)"))
    print("-" * (20 + 10 + 15))
    for i in range(1, int(prob['n']) + 1):  
        item_name = 'item ' + str(i)  
        peso = prob['pesos'][i - 1]
        valor = prob['valores'][i - 1]
        print("| {:<15} | {:<10.8f} | {:<10.8f} |".format(item_name, peso, valor))
    print("-" * (20 + 10 + 15))
    
    # pop é um dicionário que armazena a informação relacionada com a população e os indivíduos que a constituem
    # 'main' é uma lista que armazena a população principal
    # 'fitness' guarda o desempenho de cada indivíduo na população principal
    # 'temp' guarda a população temporária para reprodução
    # 'besti' é o índice do melhor indivíduo
    # 'average' é a média dos desempenhos da população atual
    pop = {'main': [], 'fitness': [], 'temp': [], 'besti': 0, 'average': 0}
    
   
    make_pop(pop, pop_size, prob)
    
    
    # Imprime os dados formatados em colunas
    print("Criação da população inicial")
    print("-" * (20 + 40  + 25))
    print("| {:<15} | {:<40} | {:<21}|".format("Individuo", "Genes", "Fitness"))
    print("-" * (20 + 40 + 25))
   
    for i, ind in enumerate(pop['main']):
        genes_str = ' '.join(map(str, ind))
        print("| {:<15} | {:<40} | {:<20} |".format(i, genes_str, pop['fitness'][i]))
    print("-" * (20 + 40 + 25))  
    
    melhores_fitness = []  # Para armazenar o melhor fitness de cada geração
    media_fitness = []   # Para armazenar a média do fitness de cada geração               
    
   # Loop principal do algoritmo genético
    for geracao in range(n_gen):
       select(pop)  
       reproduce(pop, prob)  
       replace(pop)  
       evaluate(pop, prob)
       
       print(f"gen.: {geracao + 1} best: {pop['besti']} best: {pop['fitness'][pop['besti']]:.3f} average: {pop['average']:.3f}")

       melhores_fitness.append(pop['fitness'][pop['besti']])
       media_fitness.append(pop['average'])
       
       
    geracoes = list(range(1, n_gen + 1))  
    plt.figure(figsize=(10, 5))  
    plt.plot(geracoes, melhores_fitness, label='Melhor Fitness', color='blue')
    plt.plot(geracoes, media_fitness, label='Fitness Médio', color='orange')
    plt.title('Desempenho do AG ao Longo das Gerações')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True) 
    plt.show()


          
   
def make_prob(file, prob):
    """Funcao que le os dados de um novo problema a partir de um ficheiro de texto file
     Os dados estao no formato: N L \n Peso1 Valor1 \n Peso2 Valor2 \n ... \n PesoN ValorN
     Devem ser lidos para o dicionario prob a partir do ficheiro file"""
   
    prob['pesos'] = []
    prob['valores'] = []
   
    with open(file, 'r') as f:
        prob['n'] = int(f.readline().strip())
        prob['l'] = float(f.readline().strip())
        
        # Imprime n e L
        print(f"Número de itens (N): {prob['n']}")
        print(f"Limite peso (L): {prob['l']}")


        # Lê as linhas restantes (itens)
        for i in range(prob['n']):
            line = f.readline().strip()
            peso, valor = map(float, line.split())
            prob['pesos'].append(peso)
            prob['valores'].append(valor)  

    return prob 
    
  
   
def make_pop(pop, pop_size, prob):
    """ Funcao que recebe uma populacao e cria e inicializa os seus individuos
    Estes são listas de 0 e 1 de tamanho size armazenadas em pop['main']
    Para evitar ultrapassar o limite de peso em todos os individuos iniciais, apenas um terço
    dos itens deve ser colocado a 1
    A lista pop['fitness'] deve ser inicializada a 0 para cada individuo"""    
    
    pop['main'] = []
    pop['fitness'] = []
    
    if 'pesos' in prob:
        num_ones = len(prob['pesos']) // 3  # um terço dos itens deve ser colocado a 1
        
        for _ in range(pop_size):
            ind = [0] * len(prob['pesos'])  # Inicializar o indivíduo com todos os itens não selecionados
            
            # Selecionar aleatoriamente um terço dos itens para serem 1
            ones_indices = random.sample(range(len(prob['pesos'])), num_ones)
            for idx in ones_indices:
                ind[idx] = 1
            
            pop['main'].append(ind)
            pop['fitness'].append(0)     
    
    evaluate(pop, prob)

    # Ordenar os indivíduos com base em sua aptidão, do melhor para o pior
    pop['main'] = [ind for _, ind in sorted(zip(pop['fitness'], pop['main']), reverse=True)]
    pop['fitness'] = sorted(pop['fitness'], reverse=True)

     
     
    
def evaluate(pop, prob):
    
    """ Funcao que executa o ciclo principal de avaliacao no ga.
    Para cada individuo chama a funcao de avaliacao e armazena
    o resultado na lista fitness na posicao correspondente ao individuo.
    Deve ainda calcular a media do desempenho, que deve ser armazenada
    no campo average da populacao, e guardar a posicao do melhor 
    individuo em pop['besti']. """
   
   
    bestFitness = float('-inf') # Melhor aptidão inicializada com -infinito
    totalFitness = 0  
    for i, ind in enumerate(pop['main']):
        fit = fitness(ind, prob)    
        pop['fitness'] [i] = fit
        
        totalFitness += fit
        
        if fit > bestFitness:
            bestFitness = fit
            pop['besti'] = i
     
     # Calcula a média da aptidão e armazena no campo average da população       
    pop['average'] = totalFitness / len(pop['main'])
            

       
def fitness(ind, prob):
    
    """Funcao de avaliacao: recebe um individuo e devolve uma medida
    real do seu desempenho. Quanto maior o valor melhor é o individuo.
    ind (dict): Dicionário representando o indivíduo com genes e fitness.
        prob (dict): Dicionário representando o problema com informações sobre os itens.
    """
    
    pesoTotal = 0
    valorTotal = 0
    
    for i, gene  in enumerate(ind):
        if gene == 1:
            pesoTotal += prob['pesos'][i]
            valorTotal += prob['valores'][i]
    
    
    if pesoTotal > prob['l'] :
        return prob['l'] - pesoTotal
    
    return valorTotal
    
    
def select(pop):
    """Seleccao por torneio de tamanho 3. Tres individuos sao selecionados
    aleatoriamemte e o melhor é copidado para a populacao de reproducao. O
    processo e repetido pop_size vezes"""
    pop['temp'] = []
    
    for _ in range(len(pop['main'])):
        torneio = random.sample(pop['main'], 3)  #Seleciona 3 individos aleatoriamente
        
         # Encontra o indivíduo com a maior aptidão (fitness) no torneio
        melhorInd = max(torneio, key=lambda ind: pop['fitness'][pop['main'].index(ind)])
       
        pop['temp'].append(melhorInd)
    
def reproduce(pop, prob):
    """
    Aplica o crossover e mutação a todos os indivíduos da população temporária.
    A reprodução ocorre entre pares de indivíduos selecionados aleatoriamente,
    utilizando a probabilidade de crossover (pc) e a probabilidade de mutação (pm).

    Args:
        pop (dict): Dicionário contendo a população temporária.
    """
    next_generation = []
    for i in range(0, len(pop['temp']), 2):  # Itera em passos de dois para formar pares
        if i + 1 < len(pop['temp']):  
            ind1, ind2 = pop['temp'][i], pop['temp'][i+1]
            if random.random() < pc: 
                ind1, ind2 = crossover_2points(ind1, ind2)
            ind1 = mutation(ind1)  
            ind2 = mutation(ind2)  
            next_generation.extend([ind1, ind2])
        else:
            ind1 = mutation(pop['temp'][i])  
            next_generation.append(ind1)

    pop['temp'] = next_generation  # Atualiza a população temporária com a nova geração
    evaluate(pop, prob)  # Avaliação da nova população após crossover e mutação


                
def mutation(ind):
    """Aplica mutacao a um individuo com probabilidade pm. Caso haja mutacao
    um bit e selecionado aleatoriamente e trocado"""
    for i in range(len(ind)):
        if random.random() < pm:
             # Se ocorrer uma mutação, inverte o valor do gene (0 -> 1, 1 -> 0)
            ind[i] = 1 if ind[i] == 0 else 0
    
    return ind
            
    
                
def crossover_uniforme(ind1, ind2):
    """Recombinacao uniforme de dois individuos com probabilidade pc. Cada para de bits pode ser 
    trocado nos novos indivudos com 50% de probabilidade"""
    if random.random() < pc:
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    
    return ind1, ind2
    
    

def crossover_1point(ind1, ind2):
    """Recombinacao de um ponto de corte com probabilidade pc"""
    #Verifica se a Recombinação ocorre no ponto de corte
    if random.random() < pc:
        pontocorte = random.randint(1, len(ind1) -1)
        
        ind1[pontocorte:], ind2[pontocorte:] = ind2[pontocorte:], ind1[pontocorte:]
    
    return ind1, ind2
    

def crossover_2points(ind1, ind2):
    """Recombinacao de dois ponto de corte com probabilidade pc"""
    
   
    if random.random() < pc:
        # Escolhe aleatoriamente os pontos de corte
        pontecorte = sorted(random.sample(range(1, len(ind1)), 2))
        inicio, fim = pontecorte
        
        # Realiza a troca dos genes entre os pontos de corte
        ind1[inicio:fim], ind2[inicio:fim] = ind2[inicio:fim], ind1[inicio:fim]
    
    return ind1, ind2
   


def replace(pop):
    """Substitui a população original pela população temporária.
    O melhor indivíduo é mantido na posição 0."""
   
    pop['main'][0] = pop['main'][pop['besti']].copy()
    
   
    pop['main'][1:] = pop['temp'][1:]
    
    # Limpa a população temporária
    pop['temp'] = []
    
    


if __name__ == "__main__":
    run_ga("prob3.txt")

