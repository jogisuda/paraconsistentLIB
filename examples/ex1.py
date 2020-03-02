import paraconsistentLIB


number_of_classes=3
number_of_feature_vectors_in_class = [0] * number_of_classes #syntax for list init
number_of_feature_vectors_in_class[0]=4
number_of_feature_vectors_in_class[1]=4 
number_of_feature_vectors_in_class[2]=4	
    
dimension_of_each_feature_vector=2
    
    
c = [\
0.90,  0.12,0.88,  0.14,0.88,  0.13,0.89,  0.11, \
0.55,  0.53,0.53,  0.55,0.54,  0.54,0.56,  0.54, \
0.10,  0.88,0.11,  0.86,0.12,  0.87,0.11,  0.88]


print(BestParaconsistent(number_of_classes, number_of_feature_vectors_in_class,\
dimension_of_each_feature_vector, c) )
