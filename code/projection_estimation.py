# ==========================================================
# 2D Reconstruction – Pinhole Camera Projection
# ==========================================================
# Author: Baya Mezghani
# M2 Data Science
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Load 3D points from OBJ
# -----------------------------
def lire_obj(filename):
    """
    Load vertices from an OBJ file.
    Only reads 'v' lines (ignores faces).

    Parameters:
        filename (str): Path to the OBJ file.

    Returns:
        np.ndarray: Nx3 array of vertices.
    """
    vertices = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                x, y, z = map(float, line.strip().split()[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)


# -----------------------------
# Camera Intrinsic Matrix
# -----------------------------
def matrice_intr(fx, fy, skew, cx, cy):
    """
    Create intrinsic camera matrix K.
    """
    return np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])


# -----------------------------
# Camera Extrinsic Matrix
# -----------------------------
def matrice_extr(rx_deg, ry_deg, rz_deg, tx, ty, tz):
    """
    Create extrinsic matrix [R|t] from rotations (degrees) and translation.
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    # Combined rotation (R = Rz * Ry * Rx)
    R = Rz @ Ry @ Rx

    t = np.array([[tx], [ty], [tz]])

    return np.hstack((R, t))


# -----------------------------
# Project 3D points to 2D
# -----------------------------
def projeter_points(points_3D, P):
    """
    Project 3D points using a projection matrix P.

    Parameters:
        points_3D (np.ndarray): Nx3 array of 3D points
        P (np.ndarray): 3x4 projection matrix

    Returns:
        np.ndarray: Nx2 array of 2D points
    """
    N = len(points_3D)
    homog = np.hstack((points_3D, np.ones((N, 1))))
    proj = (P @ homog.T).T
    x = proj[:, 0] / proj[:, 2]
    y = proj[:, 1] / proj[:, 2]
    return np.vstack((x, y)).T


# -----------------------------
# Estimate projection matrix (DLT)
# -----------------------------
def estimer_M(points_3D, points_2D):
    """
    Estimate projection matrix P from 3D–2D correspondences using DLT.

    Minimum 6 correspondences are required (2 equations per point).

    Parameters:
        points_3D (np.ndarray): Nx3 array of 3D points
        points_2D (np.ndarray): Nx2 array of 2D points

    Returns:
        np.ndarray: 3x4 projection matrix P
    """
    n = points_3D.shape[0]

    # Check minimum correspondences
    if n < 6:
        raise ValueError(f"At least 6 correspondences required, but got {n}")

    A = []
    for i in range(n):
        X, Y, Z = points_3D[i]
        x, y = points_2D[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # Normalize
    return P / P[-1, -1]


# -----------------------------
#  Visualization Functions
# -----------------------------
def plot_projection(points_2D, title="2D Projection", color='blue', label='Points'):
    """
    Plot 2D points.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(points_2D[:, 0], points_2D[:, 1], s=1, c=color, label=label)
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    plt.show()


def plot_comparison(proj_true, proj_est):
    """
    Plot true vs estimated 2D projections.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(proj_true[:, 0], proj_true[:, 1], s=1, c='blue', label='True Projection')
    plt.scatter(proj_est[:, 0], proj_est[:, 1], s=1, c='red', label='Estimated Projection')
    plt.title("True (Blue) vs Estimated (Red) Projection")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    plt.show()


# -----------------------------
#  Main pipeline
# -----------------------------
if __name__ == "__main__":
    # Load and normalize 3D points
    points_3D = lire_obj("data/Wooden chair.obj")
    points_3D = (points_3D - np.mean(points_3D, axis=0)) / np.max(np.linalg.norm(points_3D, axis=1))
    print(f"{len(points_3D)} points loaded.")

    # Camera matrices
    K = matrice_intr(800, 800, 0, 320, 240)
    Rt = matrice_extr(10, 30, 0, 0, 0, 3)
    M_true = K @ Rt
    print("\nTrue Projection Matrix M:\n", M_true)

    # Project points
    points_2D = projeter_points(points_3D, M_true)
    plot_projection(points_2D, title="True 2D Projection")

    # Randomly select correspondences for DLT
    idx = np.random.choice(len(points_3D), 20, replace=False)
    corr_3D = points_3D[idx]
    corr_2D = points_2D[idx]

    # Estimate projection matrix
    M_est = estimer_M(corr_3D, corr_2D)
    print("\nEstimated Projection Matrix M_hat:\n", M_est)

    # Difference between true and estimated matrices
    diff = np.linalg.norm(M_true - M_est, ord='fro')
    print(f"\nFrobenius norm difference: {diff:.6f}")

    # Project using estimated matrix
    proj_est = projeter_points(points_3D, M_est)

    # Plot comparison
    plot_comparison(points_2D, proj_est)