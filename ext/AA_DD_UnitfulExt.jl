module AA_DD_UnitfulExt

using AlgebraicArrays
using DimensionalData
using DimensionalData:@dim
using Unitful

# using LinearAlgebra

# export VectorDimArray, MatrixDimArray, AlgebraicDimArray
# export dims, rowvector, AlgebraicArray
# export rand, randn, zeros, ones

# import AlgebraicArrays: rangedims, domaindims, AlgebraicArray
# import AlgebraicArrays: MatrixArray, VectorArray 
# import LinearAlgebra: eigen
# import Base: exp, transpose
# import Base: rand, randn, zeros, ones, fill
# import DimensionalData: dims

MatrixUnitfulDimArray = MatrixArray{T, N, A} where {T <: Quantity, N, A<:AbstractDimArray{T, N}}
VectorUnitfulDimArray = VectorArray{T, N, A} where {T <: Quantity, N, A<:AbstractDimArray{T, N}}
AlgebraicUnitfulDimArray = AlgebraicArray{T, D, N, A} where {T <: Quantity, D, N, A<:AbstractDimArray{T, N}}

MatrixDimArray = MatrixArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
VectorDimArray = VectorArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
AlgebraicDimArray = AlgebraicArray{T, D, N, A} where {T, D, N, A<:AbstractDimArray{T, N}}

Base.:*(a::Unitful.Units, b::AlgebraicDimArray) = AlgebraicArray(a * parent(b), b.dims)


Base.:*(a::Unitful.Units, B::MatrixArray) = AlgebraicArray(a * Matrix(B), (rangedims(B), domaindims(B)))
Base.:*(B::Union{VectorArray,MatrixArray}, a::Unitful.Units) = a * B


end # module

