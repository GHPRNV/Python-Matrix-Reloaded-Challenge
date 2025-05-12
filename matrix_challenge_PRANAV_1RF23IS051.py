import numpy as np
import time
import tracemalloc
import cProfile

class Matrix:
    __slots__ = ['data']  # Optimization to reduce memory overhead

    def __init__(self, data):
        """
        Initialize the Matrix with a list of lists or a NumPy array.
        Accepts 1D arrays and converts them to 2D row vectors for broadcasting.
        """
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a list or NumPy array")
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Convert 1D to 2D row vector
        elif data.ndim != 2:
            raise ValueError("Only 1D or 2D matrices are allowed")
        self.data = data

    def __add__(self, other):
        """
        Add two matrices. Supports broadcasting via NumPy.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only add another Matrix")
        return Matrix(self.data + other.data)

    def __sub__(self, other):
        """
        Subtract two matrices. Supports broadcasting via NumPy.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract another Matrix")
        return Matrix(self.data - other.data)

    def __mul__(self, other):
        """
        Element-wise multiplication. Supports broadcasting via NumPy.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only multiply with another Matrix")
        return Matrix(self.data * other.data)

    def __matmul__(self, other):
        """
        Matrix multiplication (dot product) using NumPy's @ operator.
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only matrix-multiply with another Matrix")
        return Matrix(self.data @ other.data)

    def __pow__(self, power):
        """
        Raise each element to a given power using NumPy's ** operator.
        """
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be a scalar")
        return Matrix(self.data ** power)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return self.data.shape

# Task 2: Demonstrate a complex expression
def demonstrate_expression():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([5, 6])  # Treated as a row vector (1,2) and broadcasted
    result = (A + B) @ (A - B) ** 2
    print("Task 2 - Complex Expression Result:")
    print("A:\n", A)
    print("B:\n", B)
    print("(A + B) @ (A - B) ** 2:\n", result)

# Task 3: Time execution
def measure_time():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([5, 6])
    start = time.perf_counter()
    for _ in range(1000):
        result = (A + B) @ (A - B) ** 2
    end = time.perf_counter()
    print("\nTask 3 - Execution Time:")
    print(f"Time taken for 1000 iterations: {end - start:.6f} seconds")

# Task 4: Measure memory footprint
def measure_memory():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([5, 6])
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    result = (A + B) @ (A - B) ** 2
    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("\nTask 4 - Memory Footprint (Top 5 lines):")
    for stat in stats[:5]:
        print(stat)

# Task 5: Profile function calls
def compute_expression():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([5, 6])
    for _ in range(1000):
        result = (A + B) @ (A - B) ** 2
    return result

def profile_expression():
    print("\nTask 5 - cProfile Output:")
    cProfile.run('compute_expression()')

# Run all tasks
if __name__ == "__main__":
    demonstrate_expression()
    measure_time()
    measure_memory()
    profile_expression()