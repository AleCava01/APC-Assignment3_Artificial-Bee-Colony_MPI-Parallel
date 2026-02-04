import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# 1. Carica i dati
data = pd.read_csv('bees_positions.csv', header=None, 
                   names=['iter', 'id', 'x', 'y', 'z', 'w', 'best_fitness'])
#data = pd.read_csv('bees_positions.csv', header=None, 
#                   names=['iter', 'id', 'x', 'y', 'best_fitness'])

# 2. Funzione Schwefel per il "terreno" 3D
def schwefel_2d(x, y, dim=2):
    res = 418.9829 * dim
    res -= x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))
    res -= (dim - 2) * (420.9687 * np.sin(np.sqrt(np.abs(420.9687))))
    return res

# Crea la griglia per la superficie
x_range = np.linspace(-500, 500, 100)
y_range = np.linspace(-500, 500, 100)
X, Y = np.meshgrid(x_range, y_range)
Z_surf = schwefel_2d(X, Y)

# 3. Setup Plot 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Disegna la superficie semi-trasparente
surf = ax.plot_surface(X, Y, Z_surf, cmap='viridis', alpha=0.4, linewidth=0, antialiased=True)

# Inizializza le api come punti rossi 3D
# Nota: la coordinata Z delle api è il valore della loro fitness f(x, y)
scatter = ax.scatter([], [], [], c='red', s=50, edgecolors='black', depthshade=False)

# Testo informativo
title = ax.set_title('ABC 3D Convergence', fontsize=15)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')

def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

def update(frame):
    current_data = data[data['iter'] == frame]
    xs = current_data['x'].values
    ys = current_data['y'].values
    # Calcoliamo la Z (altezza) per ogni ape in base alla sua posizione
    zs = schwefel_2d(xs, ys)
    
    scatter._offsets3d = (xs, ys, zs)
    
    best_f = current_data['best_fitness'].iloc[0]
    ax.set_title(f'Iteration: {frame} | Best Fitness: {best_f:.4f}')
    
    # Facciamo ruotare leggermente la camera per un effetto dinamico
    ax.view_init(elev=30., azim=frame*0.5)
    
    return scatter,

iters = sorted(data['iter'].unique())
ani = FuncAnimation(fig, update, frames=iters, init_func=init, blit=False, interval=10)

writer = PillowWriter(fps=15)
ani.save("abc_3d_convergence.gif", writer=writer)

plt.show()