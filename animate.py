import numpy as np
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

WORLD = 100                                # must match W and H in swarm.c

data = np.loadtxt("frames.csv", delimiter=",", skiprows=1)
steps = np.unique(data[:, 0])
frames = [data[data[:, 0] == s] for s in steps]   

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, WORLD)
ax.set_ylim(0, WORLD)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

f = frames[0]
arrows = ax.quiver(f[:, 1], f[:, 2], f[:, 3], f[:, 4],
                   angles="xy", scale_units="xy", scale=1.5,
                   width=0.004, color="tab:blue")
title = ax.set_title("")

def update(k):
    f = frames[k]
    arrows.set_offsets(f[:, 1:3])         
    arrows.set_UVC(f[:, 3], f[:, 4])     
    title.set_text(f"step {int(steps[k])}")
    return arrows, title

anim = FuncAnimation(fig, update, frames=len(frames))
anim.save("swarm.gif", writer=PillowWriter(fps=20))
print("saved swarm.gif")