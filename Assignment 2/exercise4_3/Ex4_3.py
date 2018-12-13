# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:59:29 2018

@author: kamal
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
 
E_43a = [[0, 2], [2, 1], [1, 4], [4, 5], [2, 5], [2, 3], [3, 6]]
E_43b = [[0, 2], [2, 1], [1, 4], [2, 3], [3, 6]]
E_43c = [[0, 2], [2, 1], [2, 5], [5, 4], [2, 3], [3, 6]]

E_list = [E_43a, E_43b, E_43c]

x_0 = [9, 96, 83, 78, 46, 73, 64]
#x_0 = [20, 10, 15, 12, 30, 12, 15, 16]

dt = 0.001
n_vertices = 7
switch_time = 0.1 # in seconds

def left_eigen(L):
    
    l, w = linalg.eig(L, left=True, right=False)
        
    arr = np.array(l.real).tolist()
    min_index = arr.index(min(arr))
    
    print("The eigen vector corresponding to 0 eigen value is:")
    print(w[:,min_index])
    
#    for i in range(n_vertices):
#        print(w[:,i])
    print(np.matmul(w[:,min_index], x_0))
    
#    l, w = linalg.eig(L, left=True, right=False)
#    
#    for i in range(8):
#        print(w[:,i])
#    print(np.matmul(w[:,-1], x_0))

#eigen_value function
def eigen_values(L):
    
    l, w= linalg.eig(L)
#    l.sort()
    y  = list(enumerate(l))
    for i in y:
        print('{}) eigen values is: {}'.format(i[0], i[1]))
    #print("Eigen values are", l)
    #print()
    #print("The second eigen value is", l[1])


#input E_list - list of edges for n_graph
#      vertices - number of vertices
def Laplacian_list(E_list, vertices):
    L_list = []
    for i in range(len(E_list)):
        L_list.append(get_laplacian(E_list[i], vertices, False))
    return L_list

def plot(x, T):
    t = [i*0.001 for i in range(int(T/dt))]
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.plot(t, x[0], label="robot 1")
    plt.plot(t, x[1], label="robot 2")
    plt.plot(t, x[2], label="robot 3")
    plt.plot(t, x[3], label="robot 4")
    plt.plot(t, x[4], label="robot 5")
    plt.plot(t, x[5], label="robot 6")
    plt.plot(t, x[6], label="robot 7")
    #plt.plot(t, x[7], label="robot 8")
    #plt.plot(t, x[8], label="robot 9")

    legend = plt.legend(loc='upper right', shadow=True)
    #legend = plt.legend(loc='center', shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    
    plt.show()

def convergence(x, vertices):
        for i in range(vertices):
            print("The convergence for robot {}".format(i), x[i][-1])

def get_laplacian(E, n_vertices, flag):
    
    A = np.zeros((n_vertices, n_vertices)) # adjancency matrix
    D = np.zeros((n_vertices, n_vertices)) # degree matrix
    laplacian = np.zeros((n_vertices, n_vertices))

    
    if flag == False:
        # calculate the degree matrix for undirected graph
        for each_vertex in range(n_vertices): # For each_vertex we calculate the degree of that vertex
            degree = 0 

            for i in list(range(0, len(E))):
                for j in list(range(0, 2)):
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
        

def simulate_consensus(x_0, T, switch_time, L, dt=0.001):

    n_step = int(T/dt)
    x = [n_step*[0] for i in range(8)]
    c = 0
    #x_0 = [20,10,15,12,30,12,15,16,25] #state of robot at initial stage
    
    for i in range(len(x_0)):
        x[i][0] = x_0[i]
    
    
    
    for step in range(1,int(T/dt)):
        if step % (switch_time*(T/dt)) == 0:
            c +=1 
        elif step % len(L)*switch_time*(T/dt) == 0:
            c = 0
        else:
            pass
        #a 
        if c == 0:
            for r in range(n_vertices):
                if(r == 0):
                    x[0][step] = x[0][step-1] + dt*(x[2][step-1] - x[0][step-1])
                elif(r == 1):
                    x[1][step] = x[1][step-1] + dt*(x[2][step-1] + x[4][step-1] - 2*x[1][step-1])
                elif (r == 2):
                    x[2][step] = x[2][step-1] + dt*(x[1][step-1] + x[0][step-1] + x[3][step-1] + x[5][step-1] - 4*x[2][step-1])
                elif(r == 3):
                    x[3][step] = x[3][step-1] + dt*(x[2][step-1] + x[6][step-1] - 2*x[3][step-1])
                elif(r == 4):
                    x[4][step] = x[4][step-1] + dt*(x[1][step-1] + x[5][step-1] - 2*x[4][step-1])
                elif(r == 5):
                    x[5][step] = x[5][step-1] + dt*(x[4][step-1] + x[2][step-1] - 2*x[5][step-1])
                elif(r == 6):
                    x[6][step] =  x[6][step-1] + dt*(x[3][step-1] - x[6][step-1])
        


        #b
        if c == 1:
            
            for r in range(n_vertices):
                if(r == 0):
                    x[0][step] = x[0][step-1] + dt*(x[2][step-1] - x[0][step-1])
                elif(r == 1):
                    x[1][step] = x[1][step-1] + dt*(x[2][step-1] + x[4][step-1] - 2*x[1][step-1])
                elif (r == 2):
                    x[2][step] = x[2][step-1] + dt*(x[1][step-1] + x[0][step-1] + x[3][step-1] - 3*x[2][step-1])
                elif(r == 3):
                    x[3][step] = x[3][step-1] + dt*(x[2][step-1] + x[6][step-1] - 2*x[3][step-1])
                elif(r == 4):
                    x[4][step] = x[4][step-1] + dt*(x[1][step-1] + x[5][step-1] - 2*x[4][step-1])
                elif(r == 5):
                    x[5][step] = x[5][step-1] + dt*(x[4][step-1] - x[5][step-1])
                elif(r == 6):
                    x[6][step] =  x[6][step-1] + dt*(x[3][step-1] - x[6][step-1])
            
        #c
        if c == 1:
            for r in range(n_vertices):
                if(r == 0):
                    x[0][step] = x[0][step-1] + dt*(x[2][step-1] - x[0][step-1])
                elif(r == 1):
                    x[1][step] = x[1][step-1] + dt*(x[2][step-1] - x[1][step-1])
                elif (r == 2):
                    x[2][step] = x[2][step-1] + dt*(x[1][step-1]+ x[0][step-1] + x[3][step-1] + x[5][step-1] - 4*x[2][step-1])
                elif(r == 3):
                    x[3][step] = x[3][step-1] + dt*(x[2][step-1] + x[6][step-1] - 2*x[3][step-1])
                elif(r == 4):
                    x[4][step] = x[4][step-1] + dt*(x[5][step-1] - x[4][step-1])
                elif(r == 5):
                    x[5][step] = x[5][step-1] + dt*(x[4][step-1] - x[5][step-1])
                elif(r == 6):
                    x[6][step] =  x[6][step-1] + dt*(x[3][step-1] - x[6][step-1])
            
    plot(x, T)
    convergence(x, 7)


#   print()
#   print("The eigen values are :\n")
#   print(eigen_values(get_laplacian(E_43a, 7, False)))


simulate_consensus(x_0, 50, switch_time, Laplacian_list(E_list, n_vertices), dt)


























