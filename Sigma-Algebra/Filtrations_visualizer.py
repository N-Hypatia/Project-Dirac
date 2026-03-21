import matplotlib.pyplot as plt
import numpy as np

# 1. Génération d'un signal "réel" (mouvement brownien)
np.random.seed(42)
n_steps = 100
reals = np.cumsum(np.random.randn(n_steps))

# 2. On définit 3 niveaux de filtration (F_t)
# F_base : On ne connaît que la tendance globale (2 blocs)
# F_moyenne : On a une info tous les 10 pas
# F_fine : On voit tout le signal

def filtrer(signal, window):
    return np.repeat([np.mean(signal[i:i+window]) for i in range(0, n_steps, window)], window)

f_base = filtrer(reals, 50)
f_moyenne = filtrer(reals, 10)

plt.figure(figsize=(12, 6))
plt.plot(reals, label="Information Totale (Sigma-algèbre discrète)", alpha=0.8, color='black')
plt.step(range(n_steps), f_moyenne, label="Filtration intermédiaire (F_10)", color='blue', where='post')
plt.step(range(n_steps), f_base, label="Filtration grossière (F_50)", color='red', linewidth=2, where='post')

plt.title("Visualisation de l'augmentation de l'information (Filtration)")
plt.xlabel("Temps (t)")
plt.ylabel("Valeur de l'actif")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# Sigma-Algebra/Filtrations_visualizer.py