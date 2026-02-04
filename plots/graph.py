import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


# 1. Carico dati (aggiunta colonna 'best_fitness')
data = pd.read_csv('bees_positions.csv', header=None, 
                   names=['iter', 'id', 'x', 'y', 'z', 'w', 'best_fitness'])

# 2. Schwefel Function Background
def schwefel_2d(x, y, dim=4):
    res = 418.9829 * dim
    res -= x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))
    # Ottimo teorico per le altre dimensioni
    res -= (dim - 2) * (420.9687 * np.sin(np.sqrt(np.abs(420.9687))))
    return res

x_range = np.linspace(-500, 500, 150)
y_range = np.linspace(-500, 500, 150)
X, Y = np.meshgrid(x_range, y_range)
Z = schwefel_2d(X, Y)

# 3. Setup Plot
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
fig.colorbar(contour, label='Objective Function Value')

# Api: Rosse ('red'), più grandi (s=60) e con bordo nero per visibilità
scatter = ax.scatter([], [], c='red', edgecolors='black', s=60, label='Bees', zorder=3)

# Testo informativo in alto a sinistra
info_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, color='white',
                    fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))

ax.set_xlabel('Dimension 1 (X)')
ax.set_ylabel('Dimension 2 (Y)')

def init():
    scatter.set_offsets(np.empty((0, 2)))
    info_text.set_text('')
    return scatter, info_text

def update(frame):
    current_data = data[data['iter'] == frame]
    points = current_data[['x', 'y']].values
    
    # Prendiamo il miglior fitness di questa iterazione (è uguale per tutte le righe di quell'iter)
    best_val = current_data['best_fitness'].iloc[0]
    
    scatter.set_offsets(points)
    
    # Aggiornamento informazioni "on top"
    info_text.set_text(f'Iteration: {frame}\nBest Fitness: {best_val:.6f}\nNum Bees: {len(current_data)}')
    
    return scatter, info_text

iters = sorted(data['iter'].unique())
ani = FuncAnimation(fig, update, frames=iters, init_func=init, blit=True, interval=20)
writer = PillowWriter(fps=15)
ani.save("abc_2d_convergence.gif", writer=writer)

plt.title('ABC Swarm Behavior on Schwefel Landscape', fontsize=14)
plt.legend(loc='lower right')
plt.show()