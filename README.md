# Pinhole Camera Projection for 2D Reconstruction

This project was developed as part of the **2D Reconstruction course**.

It demonstrates the geometric relationship between **3D points and their 2D projections** using a **pinhole camera model**. The project also implements the **estimation of the camera projection matrix** from 3D–2D correspondences using the **Direct Linear Transformation (DLT)** algorithm.

The goal is to better understand how a camera maps points from **3D space to the 2D image plane** and how the projection matrix can be recovered from observed correspondences.

---

# Project Objectives

This project aims to:

* Implement a **pinhole camera model**
* Define **intrinsic and extrinsic camera parameters**
* Project **3D points onto the 2D image plane**
* Estimate the **camera projection matrix**
* Compare the **true projection matrix** with the estimated one

---

# Camera Model

The projection of a 3D point onto an image plane follows:

P = K [R | t]

Where:

* **K** : intrinsic camera matrix
* **R** : rotation matrix
* **t** : translation vector

The projection matrix **P** maps a 3D point in homogeneous coordinates to its 2D image coordinates.

---

# Implemented Components

## 1. Loading 3D Data

3D vertices are loaded from an **OBJ file**.

Function:

```
lire_obj()
```

---

## 2. Camera Intrinsic Matrix

The intrinsic matrix defines the internal parameters of the camera:

K = [[fx, s, cx],
[0, fy, cy],
[0, 0, 1]]

Implemented in:

```
matrice_intr()
```

---

## 3. Camera Extrinsic Matrix

The extrinsic parameters define the **camera pose**:

* rotation around x, y, z axes
* translation in 3D space

Implemented in:

```
matrice_extr()
```

---

## 4. 3D to 2D Projection

3D points are projected onto the image plane using:

P = K[R|t]

Implemented in:

```
projeter_points()
```

---

## 5. Projection Matrix Estimation (DLT)

The projection matrix is estimated from **3D–2D correspondences** using the **Direct Linear Transformation (DLT)** algorithm.

Steps:

1. Build the matrix **A**
2. Solve using **Singular Value Decomposition (SVD)**
3. Extract the projection matrix **P**

Implemented in:

```
estimer_M()
```

---

# Installation

Clone the repository:

```
git clone [https://github.com/Baya-Mezghani/pinhole-camera-projection.git]
cd pinhole-camera-projection
```

---

# Create a Virtual Environment

Create a Python virtual environment:

```
python -m venv venv
```

Activate the environment.

```
venv\Scripts\activate
```

---

# Install Dependencies

Install the required packages:

```
pip install -r requirements.txt
```

---

# Run the Project

Run the main script:

```
python code/projection_estimation.py
```

The program will:

* project the 3D points onto the image plane
* estimate the projection matrix
* display a comparison between the **true** and **estimated** projections

---

# Project Structure

```
pinhole-camera-projection
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── code
│   ├── projection_estimation.py
│   └── mesh_renderer.py
│
├── data
│   └── wooden_chair.obj
│
└── results
    ├── projection.png
    └── comparison.png
```

---

# Technologies Used

* Python
* NumPy
* Matplotlib

---

# Author

**Baya Mezghani** 
📧 baya.mezghani@ensi-uma.tn
