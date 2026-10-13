# AlgebraicArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/AlgebraicArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/AlgebraicArrays.jl/dev/)
[![Build Status](https://github.com/ggebbie/AlgebraicArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/AlgebraicArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/AlgebraicArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/AlgebraicArrays.jl)

# The issue

Box models and other gridded datasets can be stored in concise arrays that are organized according to physical space or some other organizational system. Geo-scientists are familiar with storing output data to file with meta-data in a self-describing format. For example, N-dimensional gridded data is naturally stored in N-dimensional arrays; for example, N=3 with latitude, longitude, and depth. N-dimensional arrays, unfortunately, are not in the right format to perform linear algebra operations with matrices and vectors. 

# A proposed solution

The Julia package `AlgebraicArrays.jl` aims to do the right thing to permit linear algebra operations and then to return the output in the same human-readable output that the investigator originally provided.

# Implementation

`VectorArray`: An N-dimensional array that acts like a vector in mathematical operations

`MatrixArray`: An N-dimensional array that acts like a matrix in mathematical operations

# Introducing a `VectorArray`

Suppose you are working with a random variable on a three-dimensional ($I \times J \times K$) grid:
```julia
I = 5; J = 4; K = 2
v = randn(I, J, K)
```
but you often need to flatten to a N-dimensional vector where $N = I*  J*  K$. Consider casting the array into a `VectorArray`. 
```julia
w = VectorArray(v)
```
It's also possible to directly construct the `VectorArray`:
```julia
rangesize = (I, J, K)
domainsize = () # set no domain dimension for a vector
w = randn((rangesize, domainsize))
```
Then perform operations as if `w` is a vector
```julia
M = 8
N = I * J * K
E = randn(M, N) # a matrix
x = E * w
```
The result `x` is a M-dimensional `Vector`.
Methods that reduce a `Vector` can now be used, i.e.,
```julia
norm(w)
dot(w,w)
transpose(w)*w
```
In addition, methods that keep the same input type will make the output a `VectorArray` if that was the input:
```julia
w.^2 
w .+ 1.0
```
There is no need to unflatten the `VectorArray` after the calculation!

To index a `VectorArray`, you may use the vector notation:
```julia
w[1]
w[N]
```
or the dimensional notation
```julia
w[1,1,1]
```
or the alternative dimensional notation:
```julia
w[(1,1,1)]
```
Caution: the `end` keyword does not currently work for the dimensional notation, i.e., `w[end,end,end]`.

Get the size of the array in vector form with `size(w)`. Get the size of dimensional array with `size(parent(w))`.

# Introducing a `MatrixArray`

A `MatrixArray` defines a mapping of a `VectorArray` from one space to a `VectorArray` in a new space. 
Suppose that we have a random mapping from a $I \times J \times K$ dimensional space to a $P \times Q$ dimensional space. This mapping can be defined by a `MatrixArray`:
```julia
P = 6 
Q = 7
domainsize = (I, J, K)
rangesize = (P, Q)
F = randn((rangesize, domainsize))
```
F is a `MatrixArray`. When a `MatrixArray` multiplies a `VectorArray`, a new `VectorArray` is produced:
```julia
y = F * w
```
The input `MatrixArray` and `VectorArray` are flattened before the calculation, and then the output `VectorArray` is unflattened. All of this takes place automatically. 

To index a `MatrixArray`, you may use the vectorized notation:
```julia
F[1, 1]
F[M, N]
```
or the dimensional notation
```julia
F[(1, 1), (1, 1, 1)]
F[(P, Q), (I, J, K)]
```

Get the size of the array in matrix form with `size(w)`. Get the size of dimensional array with `size(parent(w))`.

# Dimensional Data Extension

Both the `VectorArray` and `MatrixArray` are limiting cases of the
`AlgebraicArray` type. This package aims to define the
`AlgebraicArray` such that it composes well with packages that it
knows nothing about. For example, `DimArray`s compose well. Use the
statement

`using DimensionalData`

to load bonus code in the package-extending module `AlgebraicArraysDimensionalDataExt`. An N-dimensional state vector, `x`, can then be stored as a `VectorDimArray` where the dimensions of the array, i.e., `dims(x)` give the spatial or temporal locations.  These dimensions could also be used to denote which variable type is being referred to. The word, dimensions, unfortunately, has many different meanings. Do not confuse these dimensions with the units or physical quantities of the numerical entries in the array. The book, "Multidimensional Analysis" by George Hart deals with these physical units, instead. One may still input numerical values with units in the parent array of the `DimArray` by using the `Unitful` or `DynamicQuantities` Julia packages.

Example:
```julia
@dim YearCE "years Common Era"
@dim SurfaceRegion "surface location"
surfaceregions = ["NATL","ANT","SUBANT"]
years = (1990:1993)
m = length(years)
n = length(surfaceregions)
da = DimArray(randn(m,n),(Ti(years),SurfaceRegion(surfaceregions)))

# wrap it with an AlgebraicArray
aa = AlgebraicArray(da, (size(da),))
```

When the inner array in an `AlgebraicArray` is a `DimArray`, we can define it as a `MatrixDimArray` or `VectorDimArray`:
```julia
MatrixDimArray = MatrixArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
VectorDimArray = VectorArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
```
In this case, `aa` is a `VectorDimArray`. Access elements of `aa` with vector notation:
```julia
aa[1]
aa[length(aa)]
```
or access using the dimensional indexing with utilities from `DimensionalData.jl`:
```julia
aa[At(1990), At("NATL")]
aa[1, 3]
```

# Other methods

Methods include `*`, `exp`, `/`, `\`, `eigen`, and more.

# Other packages

Similar to `MultipliableDimArrays.jl`, this package extends linear algebra operations for `DimArray`s. This package 
replaces `MultipliableDimArrays.jl` because it 

1. works more generically with a wider range of `AbstractArray`s, and
2. does not suffer from type piracy.
