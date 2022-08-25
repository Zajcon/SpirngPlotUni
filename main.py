import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

m1 = 4
m2 = 3
f0 = 2.1
t0 = 0.0
t = 10.0
dt = 1
endTime = 10
nSteps = 500
nFrames = 100

n = int((t - t0)/dt)
print("n = " + str(n))

mac_m = np.array([
    [1/m1, 0, 0, 0],
    [0, 1/m1, 0, 0],
    [0, 0, 1/m2, 0],
    [0, 0, 0, 1/m2]
])

print(mac_m)

# Macierze zerowe
mac_x1 = np.zeros((4, n))
mac_x2 = np.zeros((4, n))
mac_f = np.zeros((4, n))

#Przypisanie 1 kolumny do macierzy
# format : x1, y1, x2, y2
mac_x1[:, 0] = 0, 0, -10, 0        # współrzędne poczatkowe dla x1 i x2
mac_x2[:, 0] = 0, 5, -3, 3         # predkosæ poczatkowa dla x1 i x2
mac_f[:, 0] = 0, -9.81 * m1, 0, -9.81 * m2      # sila poczatkowa dla x1 i x2

print(mac_x1)
#print(mac_x2)
#print(mac_f)
def equation(Y, t):
    for i in range(0,n):
        dx = mac_x1[0, i] - mac_x1[2, i]
        dy = mac_x1[1, i] - mac_x1[3, i]
        #print(dx)
        #print(dy)
        if dx != 0:
            alfa = math.atan(dy/dx)
        elif dx == 0 and dy > 0:
            alfa = sp.pi/2
        elif dx == 0 and dy < 0:
            alfa = -sp.pi/2
        else:
            alfa = 0
            s = 0
            c = 0
        s = sp.sin(alfa)
        c = sp.cos(alfa)
        #print("sin:i cos:")

        # Macierz sztywnoœci
        mac_k = np.array([
            [c**2, s*c, -c**2, -s*c],
            [s*c, s**2, -s*c, -s**2],
            [-c**2, -s*c, c**2, s*c],
            [-s*c, -s**2, s*c, s**2],
        ])

        #Obliczenia
        if i == 9:
            break
        else:
            WX2 = np.matmul(mac_m, (mac_f - np.matmul(mac_k, mac_x1))) * dt + mac_x2
            mac_x2[:, i+1] = WX2[:, 0]
            WX1 = mac_x2 * dt + mac_x2
            mac_x1[:, i+1] = WX1[:, 0]
    return WX2, WX1


    print(WX2)
    print(WX1)

Y0 = np.array([mac_x1, mac_f, mac_x2, mac_f]).reshape(160)
Y = odeint(equation, Y0, np.linspace(0, endTime, nSteps))
fig = plt.figure()

plt.plot(Y[:,0], Y[:,1])
plt.plot(Y[:,4],Y[:,5])
p1, = plt.plot(Y0[0],Y0[1], '.', markersize=10, color='blue')
p2, = plt.plot(Y0[4], Y0[5], '.', markersize=10, color='green')
spr, = plt.plot(Y0[4], Y0[5], '.', markersize=10, color='red')
plt.axes().set_aspect('equal', 'datalim')
plt.axes().grid(True)
plt.show()
