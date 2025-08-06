import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


def proje_a(e, a):
    return (np.dot(e,a)*e)


def QR(A):
    length = len(A[0])
    projs = np.zeros((length, length))
    e = np.zeros((length, length))
    u = np.zeros((length,length))
    for i in range(len(A[0])):
       u[i] = (A[:,i])
       for j in range(i):
           u[i] -= proje_a(e[j], A[:,i])
       e[i] = u[i] / np.linalg.norm(u[i])
    Q = np.array(e[:])
    Q = Q.T
    R = Q.T @ A
    Q = np.round(Q, decimals=8)
    R = np.round(R, decimals=8)

    return Q, R

def make_diagonal(H):
    H_new = H
    Q_total = np.zeros((20, 20))
    for i in range(2000):
        #os.system('cls')
        Q, R = QR(H_new)
        H_new = R @ Q
        #os.system('cls')
        #print(H_new[1][0])
        if i == 0:
            Q_total = Q
        else:
            Q_total = Q_total @ Q

    return H_new, Q_total

if __name__ == "__main__":
    dx = 1
    nodes =  100
    x = np.linspace(0 ,1, nodes + 2)
    H = np.zeros((nodes,nodes))
    for i in range(nodes):
        for j in range(nodes):
           if i == j:
                H[i][j] = 2
           if j == i + 1 or j == i - 1:
                H[i][j] = -1
    one_div_h_2 = 1/((1/(nodes+1))**2)
    H =H * one_div_h_2 *1/2
    E, psi = np.linalg.eigh(H)
    print(np.linalg.norm(psi[:,8]))
    np.set_printoptions(precision=3,  suppress=True)
    H, Q_total = make_diagonal(H)
    H = np.round(H, decimals=10)
    Q_total_norm = [[0 for _ in range(nodes)] for _ in range(nodes)]
    y = [[0 for _ in range(nodes+2)] for _ in range(nodes+2)]
    print(Q_total_norm)
    for i in range(nodes):
        norm = np.sqrt(np.sum(np.abs(Q_total[:][i]) ** 2) * (1 / (nodes + 1)))
        Q_total_norm[i] = list(Q_total[:][i]/norm)
        print(len(Q_total_norm[i]))
        Q_total_norm[i].append(0)
        Q_total_norm[i].insert(0, 0)
        for j in range(len(x)):
            y[i][j]=math.sqrt(2)*math.sin((i+1)*math.pi*x[j])
            print(y[i])
    print(Q_total_norm)
    print(len(x))
    Energy_level = 3
    plt.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=Q_total_norm[Energy_level - 1], mode='lines', name='psi'))
    fig.add_trace(go.Scatter(x=x, y=y[Energy_level - 1], mode='lines', name='psi_anal'))
    fig.update_layout(title='График psi', xaxis_title='x', yaxis_title='y')
    fig.show()
    for i in range (nodes):
        for j in range(nodes):
            if i == j:
                print(H[i][j])
    for i in range (nodes):
        print(E[i])




