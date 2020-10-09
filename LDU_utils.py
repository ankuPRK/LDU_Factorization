import numpy as np
from math import cos, sin

"""
Swaps ith and jth rows of matrix A
"""
def swap_rows(A, i, j):
    ri = A[i].copy()
    A[i] = A[j]
    A[j] = ri
    return
"""
Swaps ith and jth columns of matrix A
"""
def swap_columns(A, i, j):
    ci = A[:,i].copy()
    A[:,i] = A[:,j]
    A[:,j] = ci
    return

"""
The function finds imax = argmax over ii for abs(A_2[ii,i]) (ii=i,...,m-1) and swaps imax and ith row of P, A_2, L
and swaps i and imax columns of L

Input:
i: index of the row in which we are interested
A_2: The A_new matrix which satisfies PA = LA_new
L: The lower triangular matrix
P: Permutation matrix

Output: None
"""
def perform_swap_operation(i, A_2, L, P):
    m = A_2.shape[0]
    #get the max size element
    imax = i
    maxval = abs(A_2[i,i])
    for ii in range(i+1,m):
        if(maxval<abs(A_2[ii,i])):
            maxval = abs(A_2[ii,i])
            imax = ii
#     print("Swapping: ",i,jmax)
    swap_rows(P, i, imax)
    swap_rows(A_2, i, imax)
    """
    Corresponding change in L matrix: swap i and imax row, swap i and imax column
    (This produces the same effect as swapping just first i-1 elements of rows) 
    """
    swap_rows(L, i, imax)
    swap_columns(L, i, imax)
    return

"""
The function performs elimination operation to make A_2[i,i] = 1 and A_2[k,i] = 0 
for k>i using row subtractions. If A_2[i,i] = 0 then it leaves it intact and returns 

Input:
i: index of the row in which we are interested
A_2: The A_new matrix which satisfies PA = LA_new
L: The lower triangular matrix

Output: None
"""
def perform_elimination_operation(i, A_2, L):
    m = A_2.shape[0]
    #if max element in row is 0 then do nothing it means row is L.D.
    if A_2[i,i]*A_2[i,i] < 10e-12:
        return
    #Otherwise perform row subtraction
    for j in range(i+1, m):
#         print("accessing: ", i, j)
        coeff = A_2[j,i] / A_2[i,i]
        A_2[j] = A_2[j] - (coeff * A_2[i])
        L[j,i] = coeff
    
"""
The function performs Gaussian Elimination to return 
P: Permutation matrix, L: Lower triangular matrix with diagonals 1,
D: Diagonal matrix, U: Upper triangular matrix with diagonals 1 (some diagonals zero if rank(A) < n)

Input: A: an mXn matrix with m>=n, in form of numpy array
Output: P, L, D, U as defined above, all in form of numpy arrays
"""
def get_PA_LDU(A):
    #perform LA' = PA
    assert len(A.shape) == 2, "Only 2D matrices are allowed"
    assert A.shape[0] >= A.shape[1], "Only n_rows > n_cols matrices allowed!"
    
    A = A.astype(float)
    
    m = A.shape[0] #number of rows of A
    n = A.shape[1] #number of columns of A
    
    P = np.eye(m) #diagonal matrix
    L = np.eye(m) #diagonal matrix
    A_2 = A.copy() #initializing A'
    
    """ For every row i, perform following operations:
        -> if A_2[i,i] = 0, swap row i with row j s.t. j is min. index where A_2[j,i] != 0
        -> perform row operations to make A_2[i,j] = 0, j = i+1,i+2...m
    """
    
    for i in range(n):
        perform_swap_operation(i, A_2, L, P)
        perform_elimination_operation(i, A_2, L)

    """ We got L and P, now we need D and U from A_2 s.t. A_2 = DU.
        Initialize D = zeros(m,m), U = mxn matrix initialized with identity. For every row i, perform following operations:
        If A_2[i,i] = 0
        -> do nothing (i.e. keep that diagonal zero, and that row in U as it is)
        Else:
        -> D[i,i] = A_2[i,i]. U[i]/=A_2[i,i]
        
    """

    D = np.zeros((m,m))
    U = np.eye(A_2.shape[0], A_2.shape[1])
    
    for i in range(n):
        if A_2[i,i]*A_2[i,i] < 10e-12:
            continue
        else:
            D[i,i] = A_2[i,i]
            U[i] = A_2[i]/D[i,i]
            
    return P, L, D, U
