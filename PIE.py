from cv2 import imwrite
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve

F = 'D:\\Research_Sam\\Poisson_Image_Editing\\foreground.jpg'
B = 'D:\\Research_Sam\\Poisson_Image_Editing\\background.jpg'
M = 'D:\\Research_Sam\\Poisson_Image_Editing\\matte.png'

F_src = cv2.imread(F)
B_src = cv2.imread(B)
M_src = cv2.imread(M)

# Remember to normalize F, B, M's sizes

assert F_src.shape == B_src.shape == M_src.shape

# sz = (200, 150)
# F_src = cv2.resize(F_src, sz)
# B_src = cv2.resize(B_src, sz)
# M_src = cv2.resize(M_src, sz)

W, L, _ = F_src.shape
n = 0

omega = [[False for _ in range(L)] for _ in range(W)]
omega_coords = []
idx_dict = {}
count = 0
for i in range(W):
    for j in range(L):
        if M_src[i][j][0] == 255:
            omega[i][j] = True
            n += 1
            omega_coords.append((i, j))
            idx_dict[(i, j)] = count
            count += 1
            
nbhd_omega = []
for _ in omega_coords:
    y = _[0]
    x = _[1]
    tmp = [False, False, False, False]
    if omega[y - 1][x] == True: tmp[0] = True
    if omega[y + 1][x] == True: tmp[1] = True
    if omega[y][x - 1] == True: tmp[2] = True
    if omega[y][x + 1] == True: tmp[3] = True
    nbhd_omega.append(tmp)

div = [np.zeros(n) for _ in range(3)]
for i in range(3):
    for _ in range(n):
        x, y = omega_coords[_]
        div[i][_] = -4 * F_src[x][y][i] + F_src[x + 1][y][i] + F_src[x - 1][y][i] + F_src[x][y + 1][i] + F_src[x][y - 1][i]

# A = [np.diag(np.full(n, -4)) for _ in range(3)]
A = [sparse.lil_matrix((n, n)) for _ in range(3)]

for _ in range(3):
    for idx in range(n):
        A[_][idx, idx] = -4

b = [np.zeros(n) for _ in range(3)]

for cl in range(3):
    for i in range(n):
        b[cl][i] = div[cl][i]
        nbhd_coords = [(omega_coords[i][0] - 1, omega_coords[i][1]),\
                        (omega_coords[i][0] + 1, omega_coords[i][1]),\
                        (omega_coords[i][0], omega_coords[i][1] - 1),\
                        (omega_coords[i][0], omega_coords[i][1] + 1)]
        if all(nbhd_omega[i]):
            for _ in range(4):
                A[cl][i, idx_dict[nbhd_coords[_]]] = 1
        else:
            for _ in range(4):
                if nbhd_omega[i][_] == False:
                    b[cl][i] -= B_src[nbhd_coords[_]][cl]
                else:
                    A[cl][i, idx_dict[nbhd_coords[_]]] = nbhd_omega[i][_] * 1

for _ in range(3):
    A[_] = A[_].tocsc()

X = [[] for _ in range(3)]
for _ in range(3):
    # X[_] = np.linalg.solve(A[_], b[_])
    X[_] = spsolve(A[_], b[_])
    

new = np.zeros((W, L, 3))
for _ in range(3):
    new[:, :, _] = B_src[:, :, _]

for _ in range(n):
    for cl in range(3):
        x, y = omega_coords[_]
        new[x][y][cl] = min(255, X[cl][_])

new = np.array(new, dtype = 'uint8')
cv2.imwrite('D:\\Research_Sam\\Poisson_Image_Editing\\Output_Sam.jpg', new)
