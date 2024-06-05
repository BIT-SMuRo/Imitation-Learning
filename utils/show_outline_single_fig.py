import pathlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from getOutline import get_outline_points_from_angle


path_current = pathlib.Path(__file__).parent

file_name = "policy_test.csv"
path_load = path_current / "demo" / file_name
print(f"path_load: {path_load}")
if path_load.suffix == ".npy":
    data = np.load(path_load)
elif path_load.suffix == ".csv":
    data = pd.read_csv(path_load, header=None).values
print(f"shape: {data.shape}")
num_frame = data.shape[0]
jp, spine, outline = get_outline_points_from_angle(
    joint_angles=data,
    num_interp_points=100,
    inversion=False,
    degree=False,
    pitch_inversion=True,
    head=True,
)


def show_point(
    points,
    color_point="r",
    color_line="b",
    line_width=4,
    line_alpha=0.8,
    point_width=6,
    point_alpha=0.5,
):
    if points.ndim == 3:
        num_frame = points.shape[0]
        for i_frame in range(num_frame):
            x = points[i_frame, :, 0]
            y = points[i_frame, :, 1]
            z = points[i_frame, :, 2]
            ax.scatter(
                x,
                y,
                z,
                c=color_point,
                marker="o",
                linewidths=point_width,
                alpha=point_alpha,
            )
            ax.plot(x, y, z, c=color_line, linewidth=line_width, alpha=line_alpha)
    elif points.ndim == 2:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ax.scatter(
            x,
            y,
            z,
            c=color_point,
            marker="o",
            linewidths=point_width,
            alpha=point_alpha,
        )
        ax.plot(x, y, z, c=color_line, linewidth=line_width, alpha=line_alpha)


azim = 45
elev = 30
data_show = jp
k = 1000  #! 原单位是m，变为mm显示

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
colors = cm.get_cmap("magma_r", num_frame).colors

for i in range(int(num_frame)):
    data_one = data_show[i]
    show_point(data_one * k, color_point=colors[i], color_line=colors[i])

ax.axis("equal")
plt.grid()

bwith = 1.5
ax = plt.gca()
ax.spines["bottom"].set_linewidth(bwith)
ax.spines["left"].set_linewidth(bwith)
ax.spines["top"].set_linewidth(bwith)
ax.spines["right"].set_linewidth(bwith)

# font_label_e = {"family": "Times New Roman", "size": 20, "weight": "bold"}
# ax.set_xlabel("$X (mm)$", fontdict=font_label_e)
# ax.set_ylabel("$Y (mm)$", fontdict=font_label_e)
# ax.set_zlabel("$Z (mm)$", fontdict=font_label_e)

ax.set_xlim(-0.02 * k, 0.18 * k)
ax.set_ylim(-0.18 * k, 0.18 * k)
ax.set_zlim(-0.02 * k, 0.1 * k)

x_tick = ax.get_xticklabels()
[x_tick_temp.set_fontname("Times New Roman") for x_tick_temp in x_tick]
y_tick = ax.get_yticklabels()
[y_tick_temp.set_fontname("Times New Roman") for y_tick_temp in y_tick]
z_tick = ax.get_zticklabels()
[z_tick_temp.set_fontname("Times New Roman") for z_tick_temp in z_tick]

x_tick = ax.get_xticklabels()
[x_tick_temp.set_fontweight("bold") for x_tick_temp in x_tick]
y_tick = ax.get_yticklabels()
[y_tick_temp.set_fontweight("bold") for y_tick_temp in y_tick]
z_tick = ax.get_zticklabels()
[z_tick_temp.set_fontweight("bold") for z_tick_temp in z_tick]

plt.tick_params(labelsize=24)
ax.view_init(azim=azim, elev=elev)

plt.show()
