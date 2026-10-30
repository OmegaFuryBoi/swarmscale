import numpy as np
import matplotlib
matplotlib.use("Agg")                   
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

WORLD = None                               

data = np.loadtxt("frames.csv", delimiter=",", skiprows=1)
steps = np.unique(data[:, 0])
frames = [data[data[:, 0] == s] for s in steps]
WORLD = np.sqrt(len(frames[0]) / 0.1)        

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, WORLD)
ax.set_ylim(0, WORLD)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

def split(f):
    return f[f[:, 5] == 0], f[f[:, 5] == 1]   

prey, pred = split(frames[0])
prey_arrows = ax.quiver(prey[:, 1], prey[:, 2], prey[:, 3], prey[:, 4],
                        angles="xy", scale_units="xy", scale=1.5,
                        width=0.004, color="tab:blue")
pred_arrows = ax.quiver(pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4],
                        angles="xy", scale_units="xy", scale=0.8,
                        width=0.009, color="red")
title = ax.set_title("")

def update(k):
    prey, pred = split(frames[k])
    prey_arrows.set_offsets(prey[:, 1:3])
    prey_arrows.set_UVC(prey[:, 3], prey[:, 4])
    pred_arrows.set_offsets(pred[:, 1:3])
    pred_arrows.set_UVC(pred[:, 3], pred[:, 4])
    title.set_text(f"step {int(steps[k])}")
    return prey_arrows, pred_arrows, title

anim = FuncAnimation(fig, update, frames=len(frames))
anim.save("swarm.gif", writer=PillowWriter(fps=20))
print("saved swarm.gif")