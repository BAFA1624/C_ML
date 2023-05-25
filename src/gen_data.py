import numpy as np
import matplotlib.pyplot as plt


def f(x, m, c):
    return m * x + c


m = 2.235235
c = 9.878942
N = 100

xmin = -100
xmax = 100


data = {"m": m, "c": c, "N": N, "x": [], "y": [], "label": [], "noise": []}
data["x"] = np.array(
    [np.random.rand() * np.random.randint(xmin, xmax) for _ in range(N)]
)
data["label"] = np.array([f(x, m, c) for x in data["x"]])
data["noise"] = np.array(
    [(0.2 - 0.5 * 0.2) * y * np.random.rand() for y in data["label"]]
)
data["y"] = np.array([y + n for y, n in zip(data["label"], data["noise"])])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_xlim(-10, 10)
ax1.plot(data["x"], data["y"], "kx")
ax1.plot(data["x"], data["label"], "r-")
ax2.plot(data["x"], data["noise"])

plt.show()
