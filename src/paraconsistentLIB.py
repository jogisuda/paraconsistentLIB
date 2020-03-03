"""
This is the python version for using the Paraconsistent Analysis.
For a feature vector of length n, the function ParaconsistentAnalysis()
calculates every possible combination of features from n, and then returns
the best possible vector, that is, the set of features closest to P(1,0) in the Paraconsistent plane.

For more info, see the article from Guido, R.C; "Paraconsistent feature engineering", 
IEEE Signal Process Mag., vol. 36, no. 1, pp. 154-158, 2019
"""

#mean_similarities(double**,int,int);#vectors, of vectors, dimension

import numpy as np
import copy
from copy import deepcopy


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


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
    number_of_feature_vectors_in_class, dimension_of_each_feature_vector, c):

    '''
    Example of usage: (IMPORTANT: array c MUST BE FLATTENED OR RESHAPED TO (1, -1))
    
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
    print("\nBETA: %.3f",beta)
    print("\nP=(G1,G2)=(%.3f,%.3f)",alpha-beta,alpha+beta-1)
    print("\nDistance from P to (1,0): %.3f",np.sqrt(pow((alpha-beta)-1,2)+pow(alpha+beta-1,2)))
    print("\n\n")
    return np.sqrt(pow((alpha-beta)-1,2)+pow(alpha+beta-1,2)) 

######################/
######################

######################/
######################



def BestParaconsistent(number_of_classes, number_of_feature_vectors_in_class,\
dimension_of_each_feature_vector, c):

    featuresIndex = [Y for Y in range(dimension_of_each_feature_vector)] #representa o índice
    #de cada característica no vetor de carac.
    bestIndex = [] #as melhores carac. selecionadas
    bestDist = 2 # melhor distancia correspondente ao conjunto dos
    # melhores índices começa em mais infinito.
    for a in range(dimension_of_each_feature_vector):
        old = copy.deepcopy(bestIndex)
        for i in featuresIndex:
            if len(bestIndex) == 0 or a == 0: #se lista ainda vazia, ou ainda é a primeira passada
                #doSomething
                newC = []
                number_of_feature_vectors_in_class_AUX = [1] * number_of_classes
                count = 0
                for j in range(len(c)):
                    if j % dimension_of_each_feature_vector == i:
                        newC.append(c[j])
                dist = ParaconsistentAnalysis(number_of_classes, number_of_feature_vectors_in_class, 1, newC)
                if dist < bestDist:
                    bestDist = dist
                    if len(bestIndex) == 0:
                        bestIndex.append(i)
                    else:
                        bestIndex.pop()
                        bestIndex.append(i)
            else:
                if i not in bestIndex:
                    bestIndex.append(i) #temporariamente
                    newC = []
                    for j in range(len(c)):
                        if (j % dimension_of_each_feature_vector) in bestIndex:
                            newC.append(c[j])
                    dist = ParaconsistentAnalysis(number_of_classes, number_of_feature_vectors_in_class, len(bestIndex), newC)
                    bestIndex.remove(i) #retira o ultimo elemento colocado
                    if dist < bestDist and old == bestIndex:
                        bestDist = dist
                        bestIndex.append(i)
                    elif dist < bestDist and old != bestIndex:
                        bestDist = dist
                        bestIndex.pop()
                        bestIndex.append(i)
        bestIndex.sort()


    return bestIndex, bestDist
    


