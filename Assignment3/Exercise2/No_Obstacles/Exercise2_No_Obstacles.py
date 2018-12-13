# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:50:04 2018

@author: kamal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mp
import math
import IPython

np.set_printoptions(linewidth=100)

pi = [0, 0, -0.2, -0.2, 0.2, 0, 0, 0.1] #initial position
des = [9.5, 9.5, 10.5 ,9.5, 10.5, 10.5, 9.5, 10.5]
E = [[0, 1], [1, 2], [2, 3], [3, 0]] #Edges
obs = []
t = 20 #time
dt = 0.001 #time step
n_vertices = 4
K_f = 10
K_t = 5
K_o = 10
D_t = 2 * math.sqrt(K_t)
D_f = 2 * math.sqrt(K_f)
d_max = 1.0
vel_x = [0, 0, 0, 0]
vel_y = [0, 0, 0, 0] 

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

#question 2
def x_y(pts):
    x = []
    y = []
    pts = np.reshape(pts, (4,2))
    for p in pts:
        x.append(p[0])
        y.append(p[1])
    return x, y

obst = np.zeros((4, len(obs)))

def distance(x, y, obs):
    d = []
    for i in range(4):
        for e, j in enumerate(obs):
            d = math.sqrt((x[i] - j[0])**2 + (y[i] - j[1])**2)
            obst[i][e] = d
    return obst


L = getLaplacian(E, n_vertices)
inc = getIncidenceMatrix(E, n_vertices)

x_des, y_des = x_y(des)

zx_ref = np.matmul(np.transpose(inc), x_des)
zy_ref = np.matmul(np.transpose(inc), y_des)

pos_x = [[0] * 4]
pos_y = [[0] * 4]

x_pi, y_pi = x_y(pi)
pos_x[0] = x_pi
pos_y[0] = y_pi

F_x = np.zeros((4,len(obs)))
F_y = np.zeros((4,len(obs)))

def diff_x(x_des, pos_x):
    x_d = np.array(x_des)
    p_x = np.array(pos_x)
    diff = x_d - p_x
    return np.amin(diff)

def diff_y(y_des, pos_y):
    y_d = np.array(y_des)
    p_y = np.array(pos_y)
    diff = y_d - p_y
    return np.amin(diff) 

for i in range(1, int(t/dt)):
    distance(pos_x[i-1], pos_y[i-1], obs)
    for r in range(4):
        for j in range(len(obs)):
            if obst[r][j] < 1:
                F_x[r][j] = -K_o * ((obst[r][j] - d_max)/(obst[r][j])**3) * (pos_x[i-1][r] - obs[j][0]) 
                F_y[r][j] = -K_o * ((obst[r][j] - d_max)/(obst[r][j])**3) * (pos_y[i-1][r] - obs[j][1])
    ff_x = []  
    ff_y = []
    for s in range (4):
        ff_x.append(sum(F_x[s,:]))
        ff_y.append(sum(F_y[s,:]))
    
    posx_acc = K_f*(np.matmul(inc, zx_ref) - np.matmul(L, pos_x[i-1])) - D_f*np.matmul(L, vel_x) + (K_t*min(diff_x(x_des, pos_x[i-1]), 1.)) - D_t*np.array(vel_x) 
    posy_acc = K_f*np.matmul(inc, zy_ref) - K_f*np.matmul(L, pos_y[i-1]) - D_f*np.matmul(L, vel_y) + (K_t*min(diff_y(y_des, pos_y[i-1]), 1.)) - D_t*np.array(vel_y)
    
    vel_x = vel_x + dt*posx_acc
    vel_y = vel_y + dt*posy_acc
    
    pos_x.append([0] * 4)
    pos_y.append([0] * 4)
    
    pos_x[i] = pos_x[i-1] + dt*vel_x
    pos_y[i] = pos_y[i-1] + dt*vel_y
    
pos_xf = np.asarray(pos_x)
pos_yf = np.asarray(pos_y)
pos_f  = np.zeros([int(t/dt), 8])

for i in range(4):
    pos_f[:,2*i] = pos_xf[:,i]
    pos_f[:,2*i+1] = pos_yf[:,i]

t = [i*0.001 for i in range(int(t/dt))]
for i in range(4):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title('Behavior of the robots')
    plt.plot(t, pos_f[:,2*i])
    plt.ylabel('x pos')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, pos_f[:,2*i + 1])
    plt.xlabel('time')
    plt.ylabel('y pos')
    plt.savefig('xy_position')
plt.show()

pos_plot = np.reshape(pos_f[-1], (4, 2))

for i, r in enumerate(pos_plot):
    plt.figure(2)
    plt.title('Formation')
    plt.plot(r[0], r[1], 'ro')
    plt.text(r[0], r[1], i+1, fontsize=12, horizontalalignment='left', verticalalignment='bottom')
    plt.savefig('Formation')
plt.show()    


## assume we generated a vector x of positions every 0.001 seconds 

x = np.zeros([8,20000])
x[:,0] = np.array(pos_f[-1])
pos_t = pos_f.transpose() # onspot formation

# assume no obstacles
obstacles = []

# reduce the number of points we want to display to points every 100ms
plotx = pos_t[:,0::100]

#create a video with the data
make_animation_obstacles(plotx, E, obstacles, inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video1.mp4")

#do a sceond animation with obstacles this time
obstacles = np.array([[4,0],[4,4.7],[4,8],[6,2],[7,6],[6,10],[8,4],[8,12]])
make_animation_obstacles(plotx, E, obstacles, inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video2.mp4")

