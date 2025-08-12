import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import time
import arcade
from matplotlib.animation import FuncAnimation
import os
os.environ["PATH"] += os.pathsep + r"C:\Users\Пето\Desktop\ffmpeg\bin"
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
#norm = 1
print(norm)
#print(k)
"""
class Tunnel_effect(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height)
        global nodes
        self.sprite_list = arcade.SpriteList()
        self.t = 0
        self.start_time = time.perf_counter()
        self.norm_x = 1200 / nodes
        self.norm_y = 400

    def on_draw(self):
        self.clear()
        global psi_x_t
        global x
        global nodes
        for i in range(12):
            arcade.draw_line(100*(i+1), 0,100*(i+1), 800, arcade.color.LIGHT_GRAY)
            arcade.draw_text(round(x[i*int(nodes/12)],3) , i*100+10, 16, arcade.color.WHITE, 14, "Helvetica")
            if i == 5:
                arcade.draw_line(100 * (i + 1), 0, 100 * (i + 1), 800, arcade.color.LIGHT_GRAY, 3)
        for i in range(8):
            arcade.draw_line(0, 100*(i+1),1200, 100*(i+1), arcade.color.LIGHT_GRAY)
            arcade.draw_text(-0.75+ i*0.25, 610, i * 100 + 110, arcade.color.WHITE, 14, "Helvetica")
            if i ==3:
                arcade.draw_line(0, 100*(i+1),1200, 100*(i+1), arcade.color.LIGHT_GRAY, 3)
        for i in range(nodes - 1):
            y_1 = 400 * psi_x_t[i].real  + 400
            y_2 = 400 * psi_x_t[i+1].real  + 400
            x_1 = i*1200/nodes
            x_2 = (i+1)*1200/nodes
            arcade.draw_line(x_1, y_1, x_2, y_2, arcade.color.RED)
            y_1 = 400 * psi_x_t[i].imag + 400
            y_2 = 400 * psi_x_t[i + 1].imag + 400
            x_1 = i * 1200 / nodes
            x_2 = (i + 1) * 1200 / nodes
            arcade.draw_line(x_1, y_1, x_2, y_2, arcade.color.ORANGE)
        self.sprite_list.draw()

    def on_update(self, delta):
        self.sprite_list.clear()
        global psi_x_t
        global psi_k_t_1
        global psi_k_t
        global nodes
        global V
        global dx
        #global norm
        self.t+=1
        psi_x_t *= np.exp(-1j*V*(1/2)*dt)
        psi_k_t = np.fft.fft(psi_x_t)
        psi_k_t = psi_k_t * np.exp(-1j * (k ** 2) * (1 / 2) * dt)
        psi_x_t = np.fft.ifft(psi_k_t)
        psi_x_t *= np.exp(-1j*V*(1/2)*dt)
        norm = np.sqrt(np.sum(np.abs(psi_x_t[:])**2 * dx))
        psi_x_t = psi_x_t/norm
        for i in range(nodes):
            new_rect = arcade.SpriteSolidColor(1, 1, color=arcade.color.RED)
            new_rect.center_x = i*1200/nodes
            new_rect.center_y = 2*400* psi_x_t[i].real  + 400
            self.sprite_list.append(new_rect)
        for i in range(nodes):
            new_rect = arcade.SpriteSolidColor(1, 1, color=arcade.color.ORANGE)
            new_rect.center_x = i*1200/nodes
            new_rect.center_y = 2*400 * psi_x_t[i].imag  + 400
            self.sprite_list.append(new_rect)
        #print(self.t)
        if self.t == 100:
            print(time.perf_counter() - self.start_time)
"""
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
#sim = Tunnel_effect(1200,800)
#sim.run()
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
ani.save("Tunnel.mp4", writer ="ffmpeg", fps = 60)

