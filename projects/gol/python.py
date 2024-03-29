import random
from utils import video, Timer
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def gol_naive(Nx, Ny, MAX_IT, prog, seed, *args, **kwargs):
    random.seed(seed)
    # creamos una lista de listas (matriz) con ceros
    C = [[0 for j in range(Nx)] for i in range(Ny)]
    # iteramos por cada elemento, asignando un valor aleatorio (0 ó 1)
    for i in range(Ny):
        for j in range(Nx):
            C[i][j] = random.randint(0,1)
    Cs = [C]
    Ny, Nx = len(C), len(C[0])
    t = Timer()
    t.start()
    it = tqdm(range(MAX_IT)) if prog else range(MAX_IT)
    for _ in it:
        C0 = [[0 for j in range(Nx)] for i in range(Ny)]
        for i in range(Ny):
            for j in range(Nx):
                c = C[i][j]
                v = 0
                # la primer columna no tiene vecino a la izquierda
                if j > 0:
                    v += C[i][j-1]
                # la última columna no tiene vecino a la derecha
                if j < Nx - 1:
                    v += C[i][j+1]
                # la primer fila no tiene vecino arriba
                if i > 0:
                    v += C[i-1][j]
                    # la primer columna no tiene vecino a la izquierda
                    if j > 0:
                        v += C[i-1][j-1]
                    # la última columna no tiene vecino a la derecha
                    if j < Nx - 1:
                        v += C[i-1][j+1]
                # la útlima fila no tiene vecino abajo
                if i < Ny - 1:
                    v += C[i+1][j]
                    # la primer columna no tiene vecino a la izquierda
                    if j > 0:
                        v += C[i+1][j-1]
                    # la última columna no tiene vecino a la derecha
                    if j < Nx - 1:
                        v += C[i+1][j+1]
                # nuevo estado
                if c == 0 and v == 3:
                    C0[i][j] = 1
                elif c == 1 and (v == 2 or v == 3):
                    C0[i][j] = 1
                else:
                    C0[i][j] = 0
        Cs.append(C0)
        C = C0
    t.stop()
    return Cs, t.elapsed

def gol_pad(Nx, Ny, MAX_IT, prog, seed, *args, **kwargs):
    random.seed(seed)
    C = [[0 for j in range(Nx+2)] for i in range(Ny+2)]
    for i in range(1,Ny+1):
        for j in range(1,Nx+1):
            C[i][j] = random.randint(0,1)
    Cs = [C]
    Ny, Nx = len(C)-2, len(C[0])-2
    t = Timer()
    t.start()
    it = tqdm(range(MAX_IT)) if prog else range(MAX_IT)
    for _ in it:
        C0 = [[0 for j in range(Nx+2)] for i in range(Ny+2)]
        for i in range(1, Ny+1):
            for j in range(1, Nx+1):
                # estado actual
                c = C[i][j]
                # numero vecinos
                v = C[i][j+1] + C[i][j-1] + C[i-1][j] + C[i+1][j] + \
                    C[i+1][j+1] + C[i+1][j-1] + C[i-1][j+1] + C[i-1][j-1]
                # nuevo estado
                if c == 0 and v == 3:
                    C0[i][j] = 1
                elif c == 1 and (v == 2 or v == 3):
                    C0[i][j] = 1
                else:
                    C0[i][j] = 0
        Cs.append(C0)
        C = C0
    t.stop()
    return Cs, t.elapsed

def vecinos(Nx, Ny):
    V = np.zeros((Ny*Nx, Ny*Nx))
    for j in range(Ny*Nx):
        if j > 0 and j % Nx:
            V[j][j-1] = 1
        if j < Ny*Nx - 1 and (j+1) % Nx:
            V[j][j+1] = 1
        if j >= Nx:
            V[j][j-Nx] = 1
            if (j+1) % Nx:
                V[j][j-Nx+1] = 1
            if j % Nx:
                V[j][j-Nx-1] = 1
        if j <= (Ny-1)*Nx - 1:
            V[j][j+Nx] = 1
            if j % Nx:
                V[j][j+Nx-1] = 1
            if (j+1) % Nx:
                V[j][j+Nx+1] = 1
    return V

def vecinos_sp(Nx, Ny):
    data, indices, indptr = [], [], [0]
    for i in range(Ny*Nx):
        if i > 0 and i % Nx:
            data.append(1)
            indices.append(i-1)
        if i < Ny*Nx - 1 and (i+1) % Nx:
            data.append(1)
            indices.append(i+1)
        if i >= Nx:
            data.append(1)
            indices.append(i-Nx)
            if (i+1) % Nx:
                data.append(1)
                indices.append(i-Nx+1)
            if i % Nx:
                data.append(1)
                indices.append(i-Nx-1)
        if i <= (Ny-1)*Nx - 1:
            data.append(1)
            indices.append(i+Nx)
            if i % Nx:
                data.append(1)
                indices.append(i+Nx-1)
            if (i+1) % Nx:
                data.append(1)
                indices.append(i+Nx+1)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr))

def gol_matrix(Nx, Ny, MAX_IT, prog, seed, sparse=False):
    random.seed(seed)
    np.random.seed(seed)
    C = np.random.randint(0, 2, (Ny*Nx,1))
    Cs = [C]
    V = vecinos(Nx, Ny) if not sparse else vecinos_sp(Nx, Ny)
    t = Timer()
    t.start()
    it = tqdm(range(MAX_IT)) if prog else range(MAX_IT)
    for _ in it:
        # nuevo mundo a cero
        C0 = np.zeros(C.shape)
        # calcular vecinos vivos
        N = V.dot(C)
        # aplicamos regla 1
        C0[(C == 0) & (N == 3)] = 1
        # aplicamos regla 2
        C0[(C == 1) & ((N == 2) | (N == 3))] = 1
        # guardamos resultado para visualización
        Cs.append(C0)
        C = C0
    t.stop()
    return Cs, t.elapsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Game of Life')
    parser.add_argument('-n', type=int, default=100, help='Number of cells in x and y')
    parser.add_argument('-its', type=int, default=30, help='Number of iterations')
    parser.add_argument('-o', type=str, default='output.avi', help='Output file')
    parser.add_argument('-fps', type=float, default=10.0, help='Frames per second')
    parser.add_argument('-res', type=int, default=10, help='Resolution increase factor')
    parser.add_argument('-v', action='store_true', help='Create video')
    parser.add_argument('-p', action='store_true', help='Use padded version')
    parser.add_argument('-m', action='store_true', help='Use numpy matrix')
    parser.add_argument('-s', action='store_true', help='Use numpy sparse matrix')
    parser.add_argument('-prog', action='store_true', help='Show progress bar')
    parser.add_argument('-b', action='store_true', help='Benchmark')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    print(args)
    if args.b:
        data = []
        Ns = [10, 50, 100, 200]
        for n in Ns:
            print(f"Running benchmark for {n}x{n}")
            print("Naive")
            _, elapsed1 = gol_naive(n, n, args.its, False, args.seed)
            print("Padded")
            _, elapsed2 = gol_pad(n, n, args.its, False, args.seed)
            print("Matrix")
            _, elapsed3 = gol_matrix(n, n, args.its, False, args.seed, False)
            print("Sparse")
            _, elapsed4 = gol_matrix(n, n, args.its, False, args.seed, True)
            data.append([args.its / elapsed1, args.its / elapsed2, args.its / elapsed3, args.its / elapsed4])
        df = pd.DataFrame(data, columns=['naive', 'pad', 'matrix', 'sparse'], index=Ns)
        df.to_csv('benchmark.csv')
        print(df)
        # save image with table showing value of each cell and color depending on value (similar to heatmap and confusion matrix)
        sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
        plt.savefig('benchmark.png')
        exit()
    if args.m or args.s:
        gol = gol_matrix
    elif args.p:
        gol = gol_pad
    else:
        gol = gol_naive
    frames, elapsed = gol(args.n, args.n, args.its, args.prog, args.seed, args.s)
    print(f"Elapsed time: {elapsed:0.4f} seconds")
    if args.v:
        if args.m or args.s:
            frames = [f.reshape(args.n, args.n) for f in frames]
        video(frames, args.o, args.fps, args.res)