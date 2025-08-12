import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import time
import arcade
from matplotlib.animation import FuncAnimation
import os

A = 1
k_0 = 3.9
q = 3
dt = 1/100
nodes = 2000
x = np.linspace(-70, 70, nodes)
V = np.zeros_like(x)
for i in range(0, 2000):
    if x[i]>19.5 and x[i]< 20.5:
        V[i] = 11
dx = x[1]-x[0]
print(dx)
k = np.fft.fftfreq(nodes, d = dx)*2*np.pi
psi_x_0 = A*np.exp(-(x**2)/(2*q**2))*np.exp(1j*k_0*x)
psi_k_0 = np.fft.fft(psi_x_0)
psi_k_t = psi_k_0
psi_k_t_1 = psi_k_0
psi_x_t = psi_x_0
norm = np.sqrt(np.sum(np.abs(psi_x_t ** 2) * dx))
print(norm)

def update(frame):
    global psi_x_t
    global psi_k_t_1
    global psi_k_t
    psi_x_t *= np.exp(-1j * V * (1 / 2) * dt)
    psi_k_t = np.fft.fft(psi_x_t)
    psi_k_t = psi_k_t * np.exp(-1j * (k ** 2) * (1 / 2) * dt)
    psi_x_t = np.fft.ifft(psi_k_t)
    psi_x_t *= np.exp(-1j * V * (1 / 2) * dt)
    norm = np.sqrt(np.sum(np.abs(psi_x_t[:]) ** 2 * dx))
    psi_x_t = psi_x_t/norm
    line1.set_ydata(psi_x_t.real)
    line2.set_ydata(psi_x_t.imag)
    return (line1,line2)

plt.style.use('dark_background')
plt.figure(figsize=(15,10))
plt.axvline(x=20, color='blue', linestyle='--', linewidth=4, label='Потенциальный барьер')
(line1,) = plt.plot(x, psi_x_t.real, label = "Ψ Real", color='red', linestyle='-', linewidth = 1)
(line2,) = plt.plot(x, psi_x_t.imag, label = "Ψ Imaginary", color='orange', linestyle='-', linewidth = 1)
plt.title("Ψ(x, t)")
plt.xlabel("X")
plt.ylabel("Ψ")
plt.legend()
plt.grid()
ani = FuncAnimation(plt.gcf(), update, frames=2000, interval=4, blit=True)
plt.show()
#ani.save("Tunnel.mp4", writer ="ffmpeg", fps = 60)

