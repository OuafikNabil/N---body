import numpy as np
import matplotlib.pyplot as plt

# --- 1. Initialisation ---

def init_problem(N):
    """
    Tire au sort une configuration initiale pour N corps.
    """
    masses = np.random.uniform(0.1, 3.0, size=(N,))
    positions = np.random.uniform(-10., 10., size=(2, N))
    speeds = np.random.uniform(-1., 1., size=(2, N))
    return masses, positions, speeds

def init3():
    """
    Configuration de test avec 3 corps (1 soleil, 2 planètes).
    """
    masses = np.array([3, 1, 1], dtype=float)
    positions = np.array([
        [0, 5, -5], 
        [0, 1, -1]], dtype=float)
    speeds = np.array([
        [0, -1, 1], 
        [0, 0, 0]], dtype=float)
    return masses, positions, speeds

colors3 = np.array([
    [32, 32, 32],
    (228, 90, 146),
    (111, 0, 255),
]) / 255

# --- 2. Physique (Forces et Simulation) ---

def forces(masses, positions, G=1.0):
    """
    Calcule l'ensemble des forces d'interaction gravitationnelle.
    """
    N = len(masses)
    diff = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]
    
    dist_sq = np.sum(diff**2, axis=0)
    dist_sq[np.diag_indices(N)] = 1.0  # Évite la division par zéro
    dist_cb = dist_sq ** 1.5
    
    mass_prod = masses[:, np.newaxis] * masses[np.newaxis, :]
    mag = G * mass_prod / dist_cb
    mag[np.diag_indices(N)] = 0.0 
    
    F = np.sum(diff * mag[np.newaxis, :, :], axis=2)
    return F

def simulate(masses, positions, speeds, dt=0.1, nb_steps=100):
    """
    Simule l'évolution du système sur nb_steps itérations (Euler semi-implicite).
    """
    N = len(masses)
    sim_pos = np.zeros((nb_steps, 2, N))
    
    p = positions.copy()
    s = speeds.copy()
    
    for step in range(nb_steps):
        sim_pos[step] = p.copy()
        
        F = forces(masses, p)
        a = F / masses
        s = s + a * dt
        p = p + s * dt
        
    return sim_pos

# --- 3. Affichage ---

def draw(simulation, masses, colors=None, scale=10.):
    """
    Dessine la trajectoire des corps calculés par simulate().
    """
    nb_steps, dims, N = simulation.shape
    
    if colors is None:
        colors = np.random.uniform(0.3, 1., size=(N, 3))
        
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(N):
        color = colors[i]
        x = simulation[:, 0, i]
        y = simulation[:, 1, i]
        point_size = (masses[i] * scale) ** 2
        
        ax.scatter(x, y, color=color, s=point_size, alpha=0.5, edgecolors='none')
        
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Simulation à {N} corps sur {nb_steps} itérations")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return ax

# --- 4. Exécution ---

if __name__ == "__main__":
    # Si vous êtes dans Jupyter avec %matplotlib ipympl, la figure s'affichera directement.
    # Récupération de l'état initial
    m, p, s = init3()
    
    # Simulation (on peut augmenter nb_steps pour une plus longue trace)
    sim_data = simulate(m, p, s, dt=0.1, nb_steps=100)
    
    # Dessin
    draw(sim_data, m, colors=colors3, scale=10.)
    plt.show()