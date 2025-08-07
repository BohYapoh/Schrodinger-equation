
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def make_graph(psi, level, x):
    Energy_level = level
    plt.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=psi[:, level], mode='lines', name='psi'))
    fig.update_layout(title='График psi', xaxis_title='x', yaxis_title='y')
    fig.show()


nodes = 100
x = np.linspace(-5 ,5, nodes)
H = np.zeros((nodes,nodes))
one_div_h_2 = 1/((10/(nodes-1))**2)
for i in range(nodes):
    for j in range(nodes):
        if i == j:
            H[i][j] = (1/2)*x[i]**2+ one_div_h_2
        if j == i + 1 or j == i - 1:
            H[i][j] = -one_div_h_2 *(1/2)
E, psi = np.linalg.eigh(H)
make_graph(psi, 5, x)
print(E)