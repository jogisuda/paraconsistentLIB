"""
This is the python version for using the Paraconsistent Analysis.
For a feature vector of length n, the function BestParaconsistent()
calculates every possible combination of features from n, and then returns
the best set of features (that is, the set of features closest to P(1,0) in the Paraconsistent plane) and
also the distance of the set to P(1,0).

For more info, see the article from Guido, R.C; "Paraconsistent feature engineering", 
IEEE Signal Process Mag., vol. 36, no. 1, pp. 154-158, 2019
"""

#mean_similarities(double**,int,int);#vectors, of vectors, dimension

import numpy as np
import copy
from copy import deepcopy
from itertools import chain, combinations


def mean_similarities(v, n, t):
    s = []
    for i in range(t):
    	smallest=1
    	largest=0
    	for j in range(n):
    		if v[j][i]>largest:
    			largest=v[j][i]
    		if v[j][i]<smallest:
    			smallest=v[j][i]
    	s.append(1-(largest-smallest) )
    
    m=0
    for i in range(t):
    	m+=s[i]
    m/=((float)(t))	#np..float64 ? Mais precisão		
    
    return m



def ParaconsistentAnalysis(number_of_classes, \
    number_of_feature_vectors_in_class, dimension_of_each_feature_vector, c, verbose = False):

    '''
    Example of usage:
    
    number_of_classes=3
    number_of_feature_vectors_in_class = [0] * number_of_classes #syntax for list init
    number_of_feature_vectors_in_class[0]=4
    number_of_feature_vectors_in_class[1]=4 
    number_of_feature_vectors_in_class[2]=4	
    
    dimension_of_each_feature_vector=2
    
    Example: 3 classes and 4 vectors of dimension 2 in each class
    {{0.90,0.12},{0.88,0.14},{0.88,0.13},{0.89,0.11}}   #0.88---0.90 ; 0.11---0.14
    {{0.55,0.53},{0.53,0.55},{0.54,0.54},{0.56,0.54}}   #0.53---0.56 ; 0.53---0.55
    {{0.10,0.88},{0.11,0.86},{0.12,0.87},{0.11,0.88}}   #0.10---0.12 ; 0.86---0.88	 
    
    c = [\
    0.90,0.12,0.88,0.14,0.88,0.13,0.89,0.11, \
    0.55,0.53,0.53,0.55,0.54,0.54,0.56,0.54, \
    0.10,0.88,0.11,0.86,0.12,0.87,0.11,0.88]
    '''
    #all vectors in class C_1, by all vectors in C_2, ...., by all in C_n
    
    ##############################################			
    #edit whatever you need, to the feature vectors of your problem, self line.
    #Do NOT change anything BELOW self line not !not !not 	
    ##############################################	
    
    
    #######
    C = []
    
    for i in range(number_of_classes):
        C.append([])
        C[i] = [[]]*number_of_feature_vectors_in_class[i]
    
    for i in range(number_of_classes):
        for j in range(number_of_feature_vectors_in_class[i]):
            C[i][j] = [[]] * dimension_of_each_feature_vector
    l=0
    for i in range(number_of_classes):
    	for j in range(number_of_feature_vectors_in_class[i]):
    	    for k in range(dimension_of_each_feature_vector):
    		    C[i][j][k]=c[l]
    		    l += 1
    
    #Debug info only
    #for(int i=0;i<number_of_classes;i++)
    #	for(int j=0;j<number_of_feature_vectors_in_class[i];j++)
    #		for(int k=0;k<dimension_of_each_feature_vector;k++)
    #			printf("class %d vector %d element %d is %.3f",i,j,k,C[i][j][k])
    
    Y = [0]*number_of_classes
    for i in range(number_of_classes):
    	Y[i] = mean_similarities(C[i],number_of_feature_vectors_in_class[i],dimension_of_each_feature_vector)
    alpha=Y[0]
    for i in range(number_of_classes):
    	if Y[i]<alpha:
    		alpha=Y[i]
    #print("\nALPHA: %.3f",alpha)
    smallest_range_vector_for_class = []
    for i in range(number_of_classes):
        smallest_range_vector_for_class.append([])
        smallest_range_vector_for_class[i] = [[]] * dimension_of_each_feature_vector
    
    for i in range(number_of_classes):
    	for k in range(dimension_of_each_feature_vector):
    	    smallest_range_vector_for_class[i][k]=C[i][0][k] # i k
    for i in range(number_of_classes):
        for j in range(1, number_of_feature_vectors_in_class[i]):
            for k in range(dimension_of_each_feature_vector):
                if(C[i][j][k]<smallest_range_vector_for_class[i][k]):	
                    smallest_range_vector_for_class[i][k]=C[i][j][k]
    #Debug info only
    #for(int i=0;i<number_of_classes;i++)
    #	for(int k=0;k<dimension_of_each_feature_vector;k++)
    #			print("class %d smallest component %d is %.3f",i,k,smallest_range_vector_for_class[i][k])
    
    largest_range_vector_for_class = []
    
    for i in range(number_of_classes):
        largest_range_vector_for_class.append([])
        largest_range_vector_for_class[i] = [[]] * dimension_of_each_feature_vector
    
    for i in range(number_of_classes):
    	for k in range(dimension_of_each_feature_vector):
    		largest_range_vector_for_class[i][k]=C[i][0][k]
    for i in range(number_of_classes):
        for j in range(1, number_of_feature_vectors_in_class[i]):
    	    for k in range(dimension_of_each_feature_vector):
    		    if(C[i][j][k]>largest_range_vector_for_class[i][k]):	
    			    largest_range_vector_for_class[i][k]=C[i][j][k]
    
    #Debug info only
    #for(int i=0;i<number_of_classes;i++)
    #	for(int k=0;k<dimension_of_each_feature_vector;k++)
    #			print("class %d largest component %d is %.3f",i,k,largest_range_vector_for_class[i][k]);		
    R=0
    F=0
    for ia in range(number_of_classes):
    	for ib in range(number_of_classes):
    		for j in range(number_of_feature_vectors_in_class[ib]):
    			for k in range(dimension_of_each_feature_vector):
    				if ib!=ia:
    					if (C[ib][j][k]>smallest_range_vector_for_class[ia][k])and(C[ib][j][k]<largest_range_vector_for_class[ia][k]):
    						R += 1
    					F += 1
    
    
    beta=((float)(R))/((float)(F)) #np.float(64) ? Maior prec
    
    dist = np.sqrt(pow((alpha-beta)-1,2)+pow(alpha+beta-1,2))
    if verbose == True:  
        print("BETA: %.3f",beta)
        print("P=(G1,G2)=(%.3f,%.3f)",alpha-beta,alpha+beta-1)
        print("Distance from P to (1,0): %.3f",dist)
        print("\n")
    #print("Dimensao: ", dimension_of_each_feature_vector, " -> ", dist)
    return dist 

######################/
######################

######################/
######################



def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    
def best_paraconsistent(number_of_classes, \
    number_of_feature_vectors_in_class, n_features, dataset, verbose = False):
        
    all_combinations = list(powerset(range(n_features)) )
    best_feature = -1
    best_distance = np.inf #contem a melhor distancia de cada passada. Inicializada como infinito
    best_subset = []
    
    for i in range(1, n_features+1):
        for c in all_combinations:
            #queremos apenas os vetores de tamanho i e que contenham o best_subset até agora!
            if len(c) == i and all(x in c for x in best_subset): # a elegância: contém o vazio!
                print("analisando subset {}..".format(c))
                #seleciona o dataset contendo apenas as melhores features até agora
                selected_dataset = [dataset[idx + f] for idx in range(0, len(dataset), n_features) for f in c ]
                dist = ParaconsistentAnalysis(number_of_classes, number_of_feature_vectors_in_class,
                                              len(c), selected_dataset, verbose)
                print("\t-selected ds: {} -> {}".format(selected_dataset, dist))
                if dist < best_distance:
                    best_distance = dist
                    
                    #CHECAR A SINTAXE!
                    best_feature = np.setdiff1d(c, best_subset).tolist()[0] #pega a melhor feature. CHECAR A SINTAXE!
        if best_feature != -1:
            best_subset.append(best_feature)
        else:
            return best_subset, best_distance
        #reseta para próxima passada
        best_feature = -1
        best_distance = np.inf
    return best_subset, best_distance






'''
BestsIndexes = []
BestsDists = []
number_of_classes=2
number_of_feature_vectors_in_class = [0] * number_of_classes #syntax for list init
number_of_feature_vectors_in_class[0]=(TRAIN[:, 0] == 0).sum() #saudável
number_of_feature_vectors_in_class[1]=(TRAIN[:, 0] == 1).sum() #patológico
    
dimension_of_each_feature_vector=15

X = TRAIN[:, 1:]
X[:, :] -= np.min(X[:, :], axis=0) #Scaling
X[:, :] /= np.max(X[:, :], axis=0)

c = X[:, :].flatten()
'''
print(BestParaconsistent(number_of_classes, number_of_feature_vectors_in_class,\
dimension_of_each_feature_vector, c, verbose=False) )
