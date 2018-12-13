# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:14:53 2018

@author: kamal
"""

import numpy as np
E_D = [[0,1],[0,2],[0,3],[4,0],[1, 2],[2, 3],[4, 5],[4, 6],[4, 7],[5, 6],
       [6, 7],[8, 9],[8, 10],[8, 11],[9, 10],[10, 11], [8,0]]

def eigen_values(L):
    
    l, w= np.linalg.eig(L)
    l.sort()
    print("Eigen values are", l)
    print()
    print("Two smallest eigen values are: ", l[0],l[1])
    print()
    print("Two largest eigen values are: ", l[-1],l[-2])

    return 0

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
    
L = get_laplacian(E_D, 12, True)
print("Graph laplacian for Exercise 2 is \n", L)
print()
print(eigen_values(L))
