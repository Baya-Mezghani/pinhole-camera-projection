# ==========================================================
# Pinhole Camera Mesh Renderer
# ==========================================================
# Author: Baya Mezghani
# M2 Data Science
# ==========================================================

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# -----------------------------
#  Load OBJ (verts + faces)
# -----------------------------
def lire_obj_complet(path):
    """
    Load vertices and faces from an OBJ file.

    Supports triangulation of n-gon faces.

    Returns:
        verts (np.ndarray): Nx3 array of vertices
        faces (np.ndarray): Mx3 array of triangle indices
    """
    verts = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                x, y, z = map(float, line.strip().split()[1:4])
                verts.append([x, y, z])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                idx = [int(p.split('/')[0]) - 1 for p in parts]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


# -----------------------------
# Compute vertex normals
# -----------------------------
def compute_vertex_normals(verts, faces):
    """
    Compute per-vertex normals by averaging face normals.
    """
    normals = np.zeros_like(verts)
    for f in faces:
        v0, v1, v2 = verts[f]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        normals[f] += n
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    normals /= norms[:, None]
    return normals


# -----------------------------
# Projection
# -----------------------------
def project_points(verts, K, Rt):
    """
    Project 3D points to camera image plane.
    """
    N = verts.shape[0]
    homog = np.hstack((verts, np.ones((N, 1))))
    proj = K @ (Rt @ homog.T)
    xs = proj[0, :] / proj[2, :]
    ys = proj[1, :] / proj[2, :]
    zs = (Rt @ homog.T)[2, :]
    return np.vstack([xs, ys, zs]).T


# -----------------------------
# Rasterization (simple z-buffer + Lambert shading)
# -----------------------------
def rasterize_triangles(img_w, img_h, projected, faces, vertex_normals,
                        color_base=(160, 110, 60), light_dir=np.array([0.5, 0.7, 0.3]),
                        ambient=0.25):
    """
    Rasterize triangles with z-buffer and simple Lambert shading.
    """
    color_buffer = np.zeros((img_h, img_w, 3), dtype=np.float32)
    z_buffer = np.full((img_h, img_w), -np.inf, dtype=float)
    light_dir = light_dir / np.linalg.norm(light_dir)

    for f in faces:
        pts = projected[f]  # 3x (x, y, z_cam)
        if np.any(pts[:, 2] <= 1e-6):
            continue

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        xmin, xmax = max(int(np.floor(np.min(xs))), 0), min(int(np.ceil(np.max(xs))), img_w - 1)
        ymin, ymax = max(int(np.floor(np.min(ys))), 0), min(int(np.ceil(np.max(ys))), img_h - 1)
        if xmax < 0 or ymax < 0 or xmin >= img_w or ymin >= img_h:
            continue

        # Barycentric coordinates precomputation
        v0 = np.array([xs[1] - xs[0], ys[1] - ys[0]])
        v1 = np.array([xs[2] - xs[0], ys[2] - ys[0]])
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < 1e-8:
            continue

        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                v2 = np.array([px - xs[0], py - ys[0]])
                a = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
                b = (v0[0] * v2[1] - v2[0] * v0[1]) / denom
                c = 1.0 - a - b
                if (a >= -1e-6) and (b >= -1e-6) and (c >= -1e-6):
                    z = a * zs[1] + b * zs[2] + c * zs[0]
                    if z_buffer[py, px] == -np.inf or z < z_buffer[py, px]:
                        z_buffer[py, px] = z
                        # Interpolate normals
                        n0, n1, n2 = vertex_normals[f]
                        normal = c * n0 + a * n1 + b * n2
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            normal /= norm
                        diffuse = max(np.dot(normal, light_dir), 0.0)
                        intensity = ambient + (1 - ambient) * diffuse
                        color_buffer[py, px, :] = np.array(color_base) * intensity

    return color_buffer, z_buffer


# -----------------------------
# Post-processing (blur, vignette, gamma, noise)
# -----------------------------
def postprocess_image(color_buffer, gamma=1/2.2, blur_sigma=1.0, vignette=True, noise_std=2.0):
    """
    Convert color buffer to PIL image and apply post-processing.
    """
    img = np.clip(color_buffer, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)

    # Gaussian blur
    if blur_sigma > 0:
        arr = np.array(pil).astype(np.float32)
        for c in range(3):
            arr[:, :, c] = gaussian_filter(arr[:, :, c], sigma=blur_sigma)
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Vignette effect
    if vignette:
        w, h = pil.size
        xv, yv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        mask = np.clip(1.0 - 0.5 * (xv**2 + yv**2), 0.0, 1.0)
        arr = np.array(pil).astype(np.float32)
        for c in range(3):
            arr[:, :, c] *= mask
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Gamma correction
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))

    # Add Gaussian noise
    if noise_std > 0:
        arr = np.array(pil).astype(np.float32)
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)

    return pil


# -----------------------------
#  Full render pipeline
# -----------------------------
def render_obj_to_image(path_obj,
                        img_w=640, img_h=480,
                        fx=800, fy=800, cx=320, cy=240, skew=0,
                        translation=(0, 0, 3), rotation=(0, 0, 0),
                        color_base=(160, 110, 60),
                        light_dir=np.array([0.5, 0.7, 0.3]),
                        blur_sigma=1.0):
    """
    Load OBJ, project, rasterize, and return a PIL image.
    """
    verts, faces = lire_obj_complet(path_obj)
    if len(faces) == 0:
        raise ValueError("OBJ file has no faces for rasterization.")

    # Center and normalize
    centre = np.mean(verts, axis=0)
    verts -= centre
    scale = np.max(np.linalg.norm(verts, axis=1))
    verts /= scale + 1e-12

    # Matrices
    K = np.array([[fx, skew, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    rx, ry, rz = np.deg2rad(rotation)
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    t = np.array(translation).reshape(3, 1)
    Rt = np.hstack((R, t))

    # Normals
    vnorms = compute_vertex_normals(verts, faces)

    # Projection
    projected = project_points(verts, K, Rt)

    # Rasterization
    color_buf, _ = rasterize_triangles(img_w, img_h, projected, faces, vnorms,
                                       color_base=color_base, light_dir=light_dir)

    # Post-processing
    img = postprocess_image(color_buf, blur_sigma=blur_sigma)
    return img


# -----------------------------
# Optional test run
# -----------------------------
if __name__ == "__main__":
    path = "data/Wooden chair.obj"
    img = render_obj_to_image(path_obj=path,
                              img_w=640, img_h=480,
                              fx=1000, fy=1000, cx=320, cy=240,
                              translation=(0, 0, 3.2),
                              rotation=(10, 20, 0),
                              color_base=(180, 130, 80),
                              blur_sigma=1.2)
    img.save("results/chair_rendered.jpg")
    img.show()