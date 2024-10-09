# AlgebraicArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ggebbie.github.io/AlgebraicArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ggebbie.github.io/AlgebraicArrays.jl/dev/)
[![Build Status](https://github.com/ggebbie/AlgebraicArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ggebbie/AlgebraicArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ggebbie/AlgebraicArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggebbie/AlgebraicArrays.jl)

# The issue

Box models and other gridded datasets can be stored in concise arrays that are organized according to physical space or some other organizational system. Geo-scientists are familiar with storing output data to file with meta-data in a self-describing format. For example, N-dimensional gridded data is naturally stored in N-dimensional arrays; for example, N=3 with latitude, longitude, and depth. N-dimensional arrays, unfortunately, are not in the right format to perform linear algebra operations with matrices and vectors. 

# A proposed solution

The Julia package `AlgebraicArrays.jl` aims to do the right thing to permit linear algebra operations and then returns the output in the same human-readable output that the investigator originally provided.

# Implementation

`VectorArray`: An N-dimensional array that acts like a vector in mathematical operations

`MatrixArray`: An N-dimensional array of M-dimensional arrays that acts like a matrix in mathematical operations

Thus, a algebraic matrix is stored as an array of nested arrays, where the nesting permits an easy separation for the N dimensions that are in the domain space of the matrix versus the M dimensions that are in the range space. As these dimensions are encoded in the Type information, multiple dispatch is used to extend a series of methods for these new composite types.


# Indexing

In accordance with the CR decomposition of a matrix, here we choose to store the outer dimension as the columns of the matrix (i.e., the range space), and the inner dimensions correspond to the rows of the matrix (i.e., the domain space). With this choice, a word of caution is necessary. Accessing the ith row and jth column of `A` cannot be done with `A[i,j]`. Instead one must use nested indices that are in the reverse order of typical mathematical notation `A[j][i]`.

# Dimensional Data Extension

Use the statement

`using DimensionalData`

to load bonus code in the package-extending module `AlgebraicArraysDimensionalDataExt`. An N-dimensional state vector, `x`, can then be stored as a `VectorDimArray` where the dimensions of the array, i.e., `dims(x)` give the spatial or temporal locations.  These dimensions could also be used to denote which variable type is being referred to. The word, dimensions, unfortunately, has many different meanings. Do not confuse these dimensions with the units or physical quantities of the numerical entries in the array. The book, "Multidimensional Analysis" by George Hart deals with these physical units, instead. One may still input numerical values with units in the parent array of the `DimArray` by using the `Unitful` or `DynamicQuantities` Julia packages.

# Basic methods

Methods include `*`, `exp`, `/`, and `\`.

# Other packages

Similar to `MultipliableDimArrays.jl`, this package extends linear algebra operations for `DimArray`s. This package 
replaces `MultipliableDimArrays.jl` because it 

1. works more generically with a wider range of `AbstractArray`s, and
2. does not suffer from type piracy.
