# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:36:44 2018

@author: kamal
"""

import numpy as np
def eigen_values(L, s):
    
    l, w= np.linalg.eig(L)
    l.sort()
    print("Two smallest eigen values are for graph {}:".format(s), l[0],l[1])
    print("Two largest eigen values are {} : ".format(s), l[-1],l[-2])

    return 0

#question 4 subquestion 4
def cycle_graph(n):
    E = []

    for i in range(n):
        if i != n-1:
            E.append([i,i+1])
        else:
            E.append([i,0])
    return E

#Question 4
def get_laplacian(E, n_vertices, flag):
    
    A = np.zeros((n_vertices, n_vertices)) # adjancency matrix
    D = np.zeros((n_vertices, n_vertices)) # degree matrix
    laplacian = np.zeros((n_vertices, n_vertices))

    
    if flag == False:
        # calculate the degree matrix for undirected graph
        for each_vertex in range(n_vertices): # For each_vertex we calculate the degree of that vertex
            degree = 0 

            for i in list(range(0, len(E))):
                for j in list(range(2)):
                    if(E[i][j] == each_vertex):
                        degree += 1
            D[each_vertex][each_vertex] = degree
        
        # Calculate the adjancency matrix
        for each_vertex in range(len(E)): 
            x, y= E[each_vertex][0], E[each_vertex][1]

            A[x][y] = 1 
            A[y][x] = 1 

        laplacian = np.subtract(D, A)
        return laplacian
        #print("The laplacian for undrirected graph is\n", laplacian)

    if flag == True:
        #Calculate the degree matrix for directed graph
        for each_vertex in range(n_vertices):
            degree_head = 0

            for i in list(range(len(E))):
                if(E[i][1] == each_vertex):
                    degree_head += 1
            D[each_vertex][each_vertex] = degree_head

        #Calculate the Adjancency matrix
        for each_vertex in range(len(E)):
            x = E[each_vertex][0]
            y = E[each_vertex][1]

            A[y][x] = 1
        
        laplacian = np.subtract(D, A)
        return laplacian
    
C_4 = eigen_values(get_laplacian(cycle_graph(100), 100, False), "C100")
