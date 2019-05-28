#!/usr/bin/env python3

import sys

from pprint import pprint

import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib2tikz

plt.style.use("ggplot")

filename = sys.argv[1]

raw_data = np.genfromtxt(filename, delimiter=',')
sorted_data = np.copy(raw_data)

r_values = []
p_values = []

previous_lines = []

sorted_data = sorted_data[sorted_data[:,0].argsort()]


values_same = True
i = 0
while values_same:
    p_values.append(sorted_data[i][1])
    values_same = sorted_data[i][0] == sorted_data[i+1][0]
    i += 1

sorted_data = sorted_data[sorted_data[:,1].argsort()]

values_same = True
i = 0
while values_same:
    r_values.append(sorted_data[i][0])
    values_same = sorted_data[i][1] == sorted_data[i+1][1]
    i += 1

r_values.sort()
p_values.sort()
R, P = np.meshgrid(r_values, p_values)

X, Y = R*np.cos(P), R*np.sin(P)

X = X.T
Y = Y.T

ZP = np.array([[ [float(i),0.0,0.0] for i in range(len(p_values)) ] for j in range(len(r_values)) ])
Z = np.array([[ [float(i),0.0,0.0] for i in range(len(p_values)) ] for j in range(len(r_values)) ])


print(np.shape(Z))

for row in raw_data:
    dist = row[0]
    ang = row[1]

    x_index = np.argwhere(r_values == dist)
    y_index = np.argwhere(p_values == ang)

    #print(row)
    ZP[x_index,y_index] = row[2:]

Z = ZP.copy()
pprint('Median error between HF and VQE is ' + str(np.median(Z[:,:,0] - Z[:,:,2])))

fig = plt.figure()

print(f'Shape X: {np.shape(X)}')
print(f'Shape Y: {np.shape(Y)}')
print(f'Shape Z: {np.shape(Z)}')

ax = fig.add_subplot(221,projection='3d')
ax.plot_surface(X,Y,Z[:,:,0], cmap=plt.cm.viridis)
plt.title('Hartree-Fock Energy')

#ax = fig.add_subplot(132,projection='3d')
#ax.plot_surface(X,Y,Z[:,:,1], cmap=plt.cm.YlGnBu_r)

ax = fig.add_subplot(222,projection='3d')
ax.plot_surface(X,Y,Z[:,:,2], cmap=plt.cm.viridis)
plt.title('VQE UCCSD Energy')

ax = fig.add_subplot(223)
ax.contourf(X,Y,Z[:,:,0], 100, cmap=plt.cm.viridis)
plt.title('Hartree-Fock Energy')


ax = fig.add_subplot(224)
ax.contourf(X,Y,Z[:,:,2], 100, cmap=plt.cm.viridis)
plt.title('VQE UCCSD Energy')


#fig2 = plt.figure()
#plt.plot(X[0,:],Z[0,:])i

        
def open_and_save():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.asksaveasfilename()
    matplotlib2tikz.save(str(file_path))
    
cut_through_selected = False

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, axis=%s, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.inaxes, event.xdata, event.ydata))
    clicked_axis = event.inaxes
    global cut_through_selected
    if event.xdata and event.ydata and event.button == 1:
        x = event.xdata 
        y = event.ydata
        r = max(r_values)
        ang = np.arctan2(y,x)
        closest_ang_arg = np.argmin(abs(p_values - ang))
        closest_ang = p_values[closest_ang_arg]
        print(f"Exact angle: {ang}")
        print(f"Closest angle in data: {closest_ang}")

        global cut_through_selected
        if cut_through_selected:
            clicked_axis.lines.pop(0)
            clicked_axis.lines.pop(0)

        line = clicked_axis.plot([0, r*np.cos(ang)], [0, r*np.sin(ang)], 'r')
        previous_lines.append(line)
        line = clicked_axis.plot([0, r*np.cos(closest_ang)], [0, r*np.sin(closest_ang)], 'y')
        previous_lines.append(line)
        fig.canvas.draw()
        fig.canvas.flush_events()

        cut_through_selected = True

        cut_through_fig = plt.figure()
        plt.plot(r_values, ZP[:,closest_ang_arg,2])
        plt.grid(True)
        plt.title(filename)
        plt.xlabel('Atomic seperation [Å]')
        plt.ylabel('Energy [Hartree]')
        plt.show()

    elif event.button == 3:
        if cut_through_selected:
            print("Removing lines...")
            clicked_axis.lines.pop(0)
            clicked_axis.lines.pop(0)
        cut_through_selected = False

        fig.canvas.draw()
        fig.canvas.flush_events()

    elif event.button == 2:
        open_and_save()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt._show()
