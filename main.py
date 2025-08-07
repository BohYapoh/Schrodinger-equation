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

def make_diagonal(H, iter):
    H_new = H
    Q_total = np.zeros((20, 20))
    for i in range(iter):
        Q, R = QR(H_new)
        H_new = R @ Q
        if i == 0:
            Q_total = Q
        else:
            Q_total = Q_total @ Q

    return H_new, Q_total


def make_graph(Q_total_norm, level, x):
    Energy_level = level
    plt.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=Q_total_norm[Energy_level - 1], mode='lines', name='psi'))
    fig.update_layout(title='График psi', xaxis_title='x', yaxis_title='y')
    fig.show()

if __name__ == "__main__":
    nodes = 30
    H = np.zeros((nodes,nodes))
    one_div_h_2 = 1/((1/(nodes+1))**2)
    for i in range(nodes):
        for j in range(nodes):
            if i == j:
                H[i][j] = 2
            if j == i + 1 or j == i - 1:
                H[i][j] = -1
    x = np.linspace(0, 1, nodes + 2)
    H =H * one_div_h_2 *1/2
    H, Q_total = make_diagonal(H, 1000)
    Q_total_norm = [[0 for _ in range(nodes)] for _ in range(nodes)]
    y = [[0 for _ in range(nodes+2)] for _ in range(nodes+2)]
    for i in range(nodes):
        norm = np.sqrt(np.sum(np.abs(Q_total[:][i]) ** 2) * (1 / (nodes + 1)))
        Q_total_norm[i] = list(Q_total[:][i]/norm)
        Q_total_norm[i].append(0)
        Q_total_norm[i].insert(0, 0)
        for j in range(len(x)):
            y[i][j]=math.sqrt(2)*math.sin((i+1)*math.pi*x[j])
    make_graph(Q_total_norm,3, x)
    for i in range (nodes):
        for j in range(nodes):
            if i == j:
                print(f"Уровень энергии {nodes - i} = ", H[i][j])
