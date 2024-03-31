from board import Board
from python import Python

def gol_naive(C: Board, Nx: Int, Ny: Int, MAX_IT: Int, prog, seed) -> Board:
	its = range(MAX_IT)
	for it in its:
		for i in range(Ny):
			for j in range(Nx):
				var v: SIMD[DType.uint8] = 0
				if j > 0:
					v += C[it, i, j-1]
				if j < Nx - 1:
					v += C[it, i, j+1]
				if i > 0:
					v += C[it, i-1, j]
					if j > 0:
						v += C[it, i-1, j-1]
					if j < Nx - 1:
						v += C[it, i-1, j+1]
				if i < Ny - 1:
					v += C[it, i+1, j]
					if j > 0:
						v += C[it, i+1, j-1]
					if j < Nx - 1:
						v += C[it, i+1, j+1]
				c = C[it, i, j]
				c0 = 0
				if c == 0 and v == 3:
					c0 = 1
				elif c == 1 and (v == 2 or v == 3):
					c0 = 1
				else:
					c0 = 0
				C[it+1, i, j] = c0
	return C

def main():
	var Nx: Int = 100
	var Ny: Int = 100
	var MAX_IT: Int = 100
	prog = False
	seed = 0
	# var np = Python.import_module("numpy")
	# C = np.random.randint(0, 2, (MAX_IT, Ny, Nx), dtype=np.uint8)
	var C = Board.rand(MAX_IT, Ny, Nx)
	Python.add_to_path(".")
	var utils = Python.import_module("utils")
	t = utils.Timer()
	t.start()
	C = gol_naive(C, Nx, Ny, MAX_IT, prog, seed)
	t.stop()
	print("Elapsed time: ", t.elapsed)
	utils.video(C.to_numpy())