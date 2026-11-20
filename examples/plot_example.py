import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_frame(ax, H, scale=0.15, linewidth=1.8, alpha=1.0):
    p = H[0:3, 3]
    R = H[0:3, 0:3]

    x_axis = p + scale * R[:, 0]
    y_axis = p + scale * R[:, 1]
    z_axis = p + scale * R[:, 2]

    ax.plot([p[0], x_axis[0]], [p[1], x_axis[1]], [p[2], x_axis[2]],
            color="r", linewidth=linewidth, alpha=alpha)
    ax.plot([p[0], y_axis[0]], [p[1], y_axis[1]], [p[2], y_axis[2]],
            color="g", linewidth=linewidth, alpha=alpha)
    ax.plot([p[0], z_axis[0]], [p[1], z_axis[1]], [p[2], z_axis[2]],
            color="b", linewidth=linewidth, alpha=alpha)


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: plot_example.py <data.json> <output.png>")

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    traj_pts = np.array(data["trajectory_points"])
    curve_frames = np.array(data["curve_frames"])
    h_query = np.array(data["h_query"])
    h_closest = np.array(data["h_closest"])
    curve_name = data["curve_name"]
    dist = data["distance"]
    s_star = data["s_star"]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2],
            "k--", linewidth=1.5, label="G-Polynomial Curve")
    for H in curve_frames:
        plot_frame(ax, H, scale=0.14, linewidth=1.5, alpha=0.85)

    p_query = h_query[0:3, 3]
    p_closest = h_closest[0:3, 3]

    ax.scatter(*p_query, color="darkorange", s=55, zorder=10, label="Query Pose $H$")
    plot_frame(ax, h_query, scale=0.24, linewidth=2.8, alpha=1.0)

    ax.scatter(*p_closest, color="purple", s=55, zorder=10, label=f"Closest Pose $C(s^*={s_star:.3f})$")
    plot_frame(ax, h_closest, scale=0.22, linewidth=2.5, alpha=1.0)

    ax.plot([p_query[0], p_closest[0]],
            [p_query[1], p_closest[1]],
            [p_query[2], p_closest[2]],
            color="crimson", linestyle=":", linewidth=1.8, label=f"$D(H) = {dist:.4f}$")

    ax.set_title(f"{curve_name} Shape in SE(3)", pad=14)
    ax.set_xlabel("X", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Z", labelpad=10)
    ax.legend(loc="upper right", fontsize=9)

    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    radius = max(abs(sz)) / 2.0
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, f"set_{dim}lim")(ctr - radius, ctr + radius)

    plt.tight_layout()
    plt.savefig(sys.argv[2], dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
