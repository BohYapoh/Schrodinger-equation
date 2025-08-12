
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import qr_algorihm_for_grown_men as qr

fig = go.Figure()

def make_graph(H, nodes, x, level):
    global fig
    E, psi = np.linalg.eigh(H)
    H, Q_total = qr.make_diagonal(H, 1000)
    Q_total_norm = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for i in range(nodes):
        norm = np.sqrt(np.sum(np.abs(Q_total[:, i]) ** 2) * (10 / (nodes - 1)))
        Q_total[:, i] = Q_total[:, i] / norm
        Q_total_norm[i] = Q_total[:, i].tolist()
    Q_total_norm.reverse()
    Energy_level = level
    plt.show()
    fig.add_trace(go.Scatter(x=x, y=Q_total_norm[Energy_level - 1], mode='lines', name='psi'))
    fig.update_layout(title='График psi', xaxis_title='x', yaxis_title='y')
    fig.show()
    fig = go.Figure()
    for i in range(nodes):
        Q_total_norm[i] = list(Q_total[:][i])
    for i in range(nodes):
        for j in range(nodes):
            if i == j:
                print(f"Уровень энергии {nodes - i} = ", H[i][j])

def Harmonic_oscillator(nodes, level):
    H = np.zeros((nodes,nodes))
    one_div_h_2 = 1/((10/(nodes-1))**2)
    x = np.linspace(-10, 10, nodes)
    for i in range(nodes):
        for j in range(nodes):
            if i == j:
                H[i][j] = one_div_h_2  + (1 / 2) * x[i] ** 2
            if j == i + 1 or j == i - 1:
                H[i][j] = -one_div_h_2 * (1 / 2)
    make_graph(H, nodes, x, level)


def Gauss(nodes, level):
    V_0 = 4
    H = np.zeros((nodes,nodes))
    one_div_h_2 = 1/((10/(nodes-1))**2)
    x = np.linspace(-10, 10, nodes)
    for i in range(nodes):
        for j in range(nodes):
            if i == j:
                H[i][j] = one_div_h_2  + V_0 * np.exp(-(x[i]**2)/4)
            if j == i + 1 or j == i - 1:
                H[i][j] = -one_div_h_2 * (1 / 2)
    plt.show()
    fig.add_trace(go.Scatter(x=x, y=V_0*np.exp(-(x**2)/4), mode='lines', name='Potential'))
    make_graph(H, nodes, x, level)


Harmonic_oscillator(100, 10)
#Gauss(50, 4)