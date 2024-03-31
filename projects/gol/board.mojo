from python import Python
from random import randint

# no he podido usar listas tal cual en mojo, tengo que montarme mi propia estructura
# https://docs.modular.com/mojo/notebooks/Matmul
# https://docs.modular.com/mojo/notebooks/RayTracing


alias type = DType.uint8

struct Board:
	var data: DTypePointer[type]
	var its: Int 
	var rows: Int
	var cols: Int

	# Initialize zeroeing all values
	fn __init__(inout self, its: Int, rows: Int, cols: Int):
		self.its = its
		self.rows = rows
		self.cols = cols
		self.data = DTypePointer[type].alloc(self.its * self.rows * self.cols)
		memset_zero(self.data, self.its * self.rows * self.cols)

	# Initialize taking a pointer, don't set any elements
	fn __init__(inout self, data: DTypePointer[type], its: Int, rows: Int, cols: Int):
		self.data = data
		self.its = its
		self.rows = rows
		self.cols = cols

	# Initialize with random values
	@staticmethod
	fn rand(its: Int, rows: Int, cols: Int) -> Self:
		var data = DTypePointer[type].alloc(its * rows * cols)
		randint(data, its * rows * cols, 0, 1)
		return Self(data, its, rows, cols)

	fn __getitem__(self, t: Int, y: Int, x: Int) -> Scalar[type]:
		return self.load[1](t, y, x)

	fn __setitem__(self, t: Int, y: Int, x: Int, val: Scalar[type]):
		self.store[1](t, y, x, val)

	fn load[nelts: Int](self, t:Int, y: Int, x: Int) -> SIMD[type, nelts]:
		return self.data.load[width=nelts](self._pos_to_index(t, y, x))

	fn store[nelts: Int](self, t:Int, y: Int, x: Int, val: SIMD[type, nelts]):
		return self.data.store[width=nelts](self._pos_to_index(t, y, x), val)

	fn __copyinit__(inout self, other: Self): # para poder devolerlo en funciones
		self.data = other.data
		self.its = other.its
		self.rows = other.rows
		self.cols = other.cols

	@always_inline
	fn _pos_to_index(self, it: Int, row: Int, col: Int) -> Int:
		return it * self.rows * self.cols + row * self.cols + col

	def to_numpy(self) -> PythonObject:
		var np = Python.import_module("numpy")
		var np_arr = np.zeros((self.its, self.rows, self.cols), np.uint8)
		# We use raw pointers to efficiently copy the pixels to the numpy array
		var out_pointer = Pointer(
			__mlir_op.`pop.index_to_pointer`[
				_type=__mlir_type[`!kgen.pointer<scalar<f32>>`]
			](
				SIMD[DType.index, 1](
					np_arr.__array_interface__["data"][0].__index__()
				).value
			)
		)
		var in_pointer = Pointer(
			__mlir_op.`pop.index_to_pointer`[
				_type=__mlir_type[`!kgen.pointer<scalar<f32>>`]
			](SIMD[DType.index, 1](int(self.data)).value)
		)
		for it in range(self.its):
			for row in range(self.rows):
				for col in range(self.cols):
					var index = self._pos_to_index(it, row, col)
					# out_pointer.store(
					# 	index * 3 + dim, in_pointer[index * 4 + dim]
					# )
					out_pointer.store(
						index, in_pointer.load(index)
					)
		return np_arr