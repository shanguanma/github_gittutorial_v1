#!/usr/bin/env python3

"""
function based on the matrix operation ,the matrix is 2D arrays
"""
# Indentation defaults to  two  space.
# the function(e.g._shape()) is viewed as private, in outside of module or calss cannot be used.
# It isn't also used by from module import *

def _shape(matrix):
  return list((len(matrix),len(matrix[0])))

# TODO 
def _check_not_integer(matrix):
  try:
    rows = len(matrix)
    cols = len(matrix[0])
    return True
  except TypeError:
    raise TypeError("it must be a 2D matrix")

def _verify_matrix_size(matrix_a,matrix_b):
  shape = _shape(matrix_a)
  shape += _shape(matrix_b)
  #print("shape:",shape)
  
  #if shape[0] !=shape[2] and shape[1] != shape[3]:
  #  raise ValueError (f"operands could not be broadcast together with shape."
  #                    f"({shape[0],shape[1]}),({shape[2],shape[3]})")
  
  return [shape[0],shape[2]],[shape[1],shape[2]]

def add(matrix_a,matrix_b):
  if _check_not_integer(matrix_a) and _check_not_integer(matrix_b):
    rows, cols = _verify_matrix_size(matrix_a,matrix_b)
    matrix_sum = []
    for i in range(rows[0]):
      list_sum = []
      for j in range(cols[0]):
        value_ij = matrix_a[i][j] + matrix_b[i][j]
        list_sum.append(value_ij)
      matrix_sum.append(list_sum)
    return matrix_sum

def substract(matrix_a,matrix_b):
  if _check_not_integer(matrix_a) and _check_not_integer(matrix_b):
    rows,cols = _verify_matrix_size(matrix_a,matrix_b)
    matrix_substract = []
    for i in range(rows[0]):
      list_substract = []
      for j  in range(cols[0]):
        value_ij = matrix_a[i][j] - matrix_b[i][j]
        list_substract.append(value_ij)
      matrix_substract.append(list_substract)
    return matrix_substract
def scale(matrix,n):
  if _check_not_integer(matrix):
    #if isinstance(n,int) or isinstance(n,float):
    #  pass
    #else:
    #  raise TypeError("scale number n must be int or float")
    if not (isinstance(n,int)) or (isinstance(n,float)):
      raise TypeError("scale number n must be int or float")

   
    matrix_scale = []
    # how many are rows? len(matrix) 
    for i in range(len(matrix)):
      list_ = []
      for j in range(len(matrix[0])):
        value_ij = n * matrix[i][j]
        list_.append(value_ij)
      matrix_scale.append(list_)
    return matrix_scale

# TODO:check why is IndexError: list index out of range. 
# both phalanx multiply
def multiply(matrix_a,matrix_b):
  if _check_not_integer(matrix_a) and _check_not_integer(matrix_b):
    rows, cols = _verify_matrix_size(matrix_a,matrix_b)
    if (cols[0] != rows[1]):
      raise ValueError ("first matrix colum must be equal second matrix row")
    matrix_multiply = []
    # iterate on row of the first matrix 
    for i in range(rows[0]):
      list_ = []
      # iterate on colum of the second matrix
      for j in range(cols[1]):
        val = 0
        #for k in range(cols[1]):
        # iterate on row of the second matrix
        for k in range(rows[1]):
          val = val + matrix_a[i][k] * matrix_b[k][j]
        list_.append(val) 
      matrix_multiply.append(list_)
    return matrix_multiply

def generate_multiply(matrix_a,matrix_b):
  if len(matrix_a[0]) != len(matrix_b):
    raise ValueError ("colum number of fisrt matrix  must be equal to row number of second matrix")

  res = [[0] * len(matrix_b[0]) for i in range(len(matrix_a))]
  # len(matrix_a) is row number of first matrix 
  for i in range(len(matrix_a)):
    # len(matrix_b[0]) is colum number of second matrix
    for j in range(len(matrix_b[0])):
      # len(matrix_b) is row number of second matrix
      for k in range(len(matrix_b)):
        res[i][j] += matrix_a[i][k]*matrix_b[k][j]
  return res





if __name__ == "__main__":
  matrix_a=[[1,2,3],[4,5,6]]
  matrix_b=[[2,3,4],[5,6,7]]
  matrix_c=[[1,2],[2,3],[3,4]]
  matrix_d=[[1,2],[3,4]]
  matrix_f=[[4,5],[5,6]]
  a = "name"
  print("matrix_a is: {0}\nmatrix_b is: {1}\n".format(matrix_a,matrix_b)) 
  print("they add operation result:{0}".format(add(matrix_a,matrix_b))) 
  print("they substract operation result:{0}".format(substract(matrix_a,matrix_b)))
  #print("debug matrix scaled:{0}".format(scale(matrix_a,a)))
  print("matrix scaled:{0}".format(scale(matrix_a,2)))
  #print("both phalanx  multiply:{0}".format(multiply(matrix_a,matrix_c)))
  print("both matrix multiply:{0}".format(generate_multiply(matrix_a,matrix_c)))
