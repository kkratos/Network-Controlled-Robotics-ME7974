# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:51:27 2018

@author: kamal
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mp

import IPython

np.set_printoptions(linewidth=100)

p = [1, 0, 1, 1, 0, 2, 3, 2, 4, 0] #initial position
des = [0, 2, 1, 1, 0, 0, -1, 1, 0, 1] #desired position
E = [[0, 3], [0, 4], [1, 2], [1, 4], [4, 2], [3, 4], [3, 2]] #Edges

t = 20 #time
dt = 0.001 #time step

#returns the Laplacian of graph specified by E (list of edges)
def getLaplacian(E,n_vertex):
    L = np.zeros([n_vertex,n_vertex]) #our Laplacian matrix
    Delta = np.zeros([n_vertex,n_vertex]) #this is the degree matrix
    A = np.zeros([n_vertex,n_vertex]) #this is the adjacency matrix
    for e in E: #for each edge in E
        #add degrees
        Delta[e[1],e[1]] +=1
        #add the input in the adjacency matrix
        A[e[1],e[0]] = 1
        #symmetric connection as we have undirected graphs
        Delta[e[0],e[0]] +=1
        A[e[0],e[1]] = 1
    L = Delta - A
    return L

# get incidence matrix for directed graph E (list of edges)
def getIncidenceMatrix(E,n_vertex):
    n_e = len(E)
    D = np.zeros([n_vertex,n_e])
    for e in range(n_e):
        #add the directed connection
        D[E[e][0],e] = -1
        D[E[e][1],e] = 1
    return D


def make_animation_obstacles(plotx,E,obstacles,xl=(-2,2),yl=(-2,2),inter=100, display=False):
    """
    Function to create a movie of the agents moving with obstacles
    plotx: a matrix of states ordered as (x1, y1, x2, y2, ..., xn, yn) in the rows and time in columns
    E: list of edges (each edge is a pair of vertexes)
    obstacles: a list of coordinates for each obstacle [[o1x, o1y],[o2x,o2y],...]
    xl, yl: display boundaries of the graph
    inter: interval between each point in ms
    display: if True displays the anumation as a movie, if False only returns the animation
    
    in order to keep computation quick, use in plotx points spaced by at least 100ms (and use inter=100) in this case
    """
    fig = mp.figure.Figure()
    mp.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xl, ylim=yl)
    ax.grid()

    list_of_lines = []
    for i in E: #add as many lines as there are edges
        line, = ax.plot([], [], 'o-', lw=2)
        list_of_lines.append(line)
    for obs in obstacles: #as many rounds as there are obstacles
        line, = ax.plot([obs[0]],[obs[1]],'ko-', lw=15)
        list_of_lines.append(line)

    def animate(i):
        for e in range(len(E)):
            vx1 = plotx[2*E[e][0],i]
            vy1 = plotx[2*E[e][0]+1,i]
            vx2 = plotx[2*E[e][1],i]
            vy2 = plotx[2*E[e][1]+1,i]
            list_of_lines[e].set_data([vx1,vx2],[vy1,vy2])
        return list_of_lines
    
    def init():
        return animate(0)


    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(plotx[0,:])),
        interval=inter, blit=True, init_func=init)
    plt.close(fig)
    plt.close(ani._fig)
    if(display==True):
        IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))
    return ani


def constraints(p):
    
    p_vec = np.reshape(p, (5, 2))
    D = []
    for e in E:
        r1, r2 = e[0], e[1]
        dis = (p_vec[r1][0] - p_vec[r2][0])**2 + (p_vec[r1][1] - p_vec[r2][1])**2
        D.append(dis)
    return D

def r_matrix(p):
    p_vec = np.reshape(p, (5, 2))
    mat = np.zeros((len(E), len(p)))
    
    for i, e in enumerate(E):
        r1, r2 = e[0], e[1]
        [mat[i, 2*r1], mat[i, 2*r1+1]] = 2*(p_vec[r1] - p_vec[r2])
        [mat[i, 2*r2], mat[i, 2*r2+1]] = -2*(p_vec[r1] - p_vec[r2])
    return mat

pos = [[[0]] * len(p)] 
pos[0] = p #store's at 0th position the initial position

for i in range(1, int(t/dt)):
    r_mat = r_matrix(pos[i-1]).transpose()
        
    pos.append([0 for i in range(len(p))])

    for j in range(10):
        pos[i][j] = (np.matmul(r_mat[j], np.subtract(constraints(des), constraints(pos[i-1]))))*dt + pos[i-1][j]
        
pos_plot = np.reshape(pos[-1], (5, 2))
for i, r in enumerate(pos_plot):
    plt.figure(1)
    plt.title('Formation')
    plt.plot(r[0], r[1], 'ro')
    plt.text(r[0], r[1], i+1, fontsize=12, horizontalalignment='left', verticalalignment='bottom')
#    plt.xlim(-1, 3)
#    plt.ylim(-1, 3)
    plt.savefig('formation')
plt.show()    

pos=np.asarray(pos)
pos_t=pos.transpose()

t = [i*0.001 for i in range(int(t/dt))]
for i in range(5):
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.title('Position of robots w.r.t. time')
    plt.ylabel('x pos')
    plt.plot(t, pos_t[2*i])
    
    plt.subplot(2,1,2)
    plt.xlabel('time')
    plt.ylabel('y pos')
    plt.plot(t, pos_t[2*i+1])
    plt.savefig('xy_position')
plt.show()

#set initial conditions
T = 15
# description of graph with list of edges
E = [[0, 3], [0, 4], [1, 2], [1, 4], [4, 2], [3, 4], [3, 2]]

# assume we generated a vector x of positions every 0.001 seconds 
# (here we start with random initial conditions and integrate motion at 0.5 meter/second)
dt = 0.001
x = np.zeros([10,10000])

x[:,0] = np.array(pos[-1])

for i in range(10000-1):
#    x[:,i+1] = pos_t[:,i] + dt * 0.5
    x[:,i+1] = x[:,i] + dt * 0.5

# assume no obstacles
obstacles = []

# reduce the number of points we want to display to points every 100ms
plotx = x[:,0::100]


#create a video with the data
make_animation_obstacles(plotx, E, obstacles, inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video1.mp4")


#do a sceond animation with obstacles this time
obstacles = np.array([[4,0],[4,4.7],[4,8],[6,2],[7,6],[6,10],[8,4],[8,12]])
make_animation_obstacles(plotx, E, obstacles, inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video2.mp4")

