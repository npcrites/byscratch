from typing import List
from typing import Tuple
import math
from typing import Callable

# Another type alias
Matrix = List[List[float]]

A = [[1, 2, 3], # A has 2 rows and 3 columns
    [4, 5, 6]]
""
B = [[1, 2], # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]
Vector = List[float]


def add(v:Vector, w:Vector) -> Vector:
    assert len(v) == len(w), "vectors must be same length"
    return [v_i + w_i for v_i, w_i in zip(v,w)]
assert add([1,2,3],[4,5,6]) == [5,7,9]



def subtract(v:Vector, w:Vector) -> Vector:
    assert len(v)== len(w),"vectors must be same length"
    return[v_i - w_i for v_i, w_i in zip (v,w)]
assert subtract([5,7,9],[4,5,6]) == [1,2,3]



def vector_sum(vectors: List[Vector]) -> Vector:
    #sums all corresponding elements
    #check that vectors is not empty
    assert vectors, "no vectors provided!"
    
    #check that vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    
    #the ith element of the result is the sum of every vector[i]
    return[sum(vector[i] for vector in vectors) 
               for i in range(num_elements)]

assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]



def scalar_multiply(c: float, v:Vector) -> Vector:
    return [c* v_i for v_i in v]

assert scalar_multiply(2,[1,2,3]) == [2,4,6]



def vector_mean(vectors:List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
assert vector_mean([[1,2],[3,4],[5,6]]) == [3,4]



def dot(v: Vector, w: Vector) -> float:
    assert len(v)  == len(w)
    return sum(v_i * v_w for v_i, v_w in zip(v,w))
assert dot ([1,2,3], [4,5,6]) == 32



def sum_of_squares (v: Vector)-> float:
    return dot(v,v)



def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))



def squared_distance (v:Vector, w:Vector) -> float:
    return sum_of_squares(subtract(v,w))



def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v,w))



def distance(v: Vector, w: Vector) -> float: 
    return magnitude(subtract(v,w))

from typing import Tuple

def shape(A:Matrix) -> Tuple[int, int]:
  #"Returns (# of rows of A, # of columns of A)"""
  num_rows = len(A)
  num_cols = len(A[0]) if A else 0 # number of elements in first row 
  return num_rows, num_cols

# If a matrix has n rows and k columns, we will refer to it as an n × k matrix. 
# We can (and sometimes will) think of each row of an n × k matrix as a vector 
# of length k, and each column as a vector of length n:

def get_row(A: Matrix, i: int) -> Vector:
  ##"""Returns the i-th row of A (as a Vector)"""
  return A[i] # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector: 
  ##"""Returns the j-th column of A (as a Vector)""" 
  return [A_i[j] for A_i in A] # jth element of row A_i
  # for A_i in A] # for each row A_i
    

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
#"""Returns a num_rows x num_cols matrix whose (i,j)-th entry is entry_fn(i, j)
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)] # [entry_fn(i, 0), ... ] for i in range(num_rows)] # create one list for each i
 
    
def identity_matrix(num_diag):
    return make_matrix(num_diag, num_diag, lambda i,j:1 if i ==j else 0) 
  


# Add these tests to the end of your linear_algebra.py file
# commit and push the update to GitHub

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Extra assert statements to test all of the functions you will need
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9], "add not working"
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3], "subtract not working"
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3
assert magnitude([3, 4]) == 5
assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns
assert distance([1,1],[4,1]) == 3.0
assert squared_distance([1,2,3],[2,3,4]) == 3
assert scalar_multiply(2, [1,2,3]) == [2,4,6]
assert magnitude([0,0,4,3]) == 5.0

# Work on an Identity Matrix
id = [  [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1] ]

assert get_column(id,2) == [0, 0, 1, 0, 0]
assert get_row(id,2) == [0, 0, 1, 0, 0]
assert get_column(id,2) == get_row(id,2)
assert identity_matrix(5) == id
assert make_matrix(5,5, lambda i,j: 1 if i == j else 0) == id
assert shape(id) == (5,5)
            


