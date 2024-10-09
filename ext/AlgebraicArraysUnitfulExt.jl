module AlgebraicArraysUnitfulExt

using AlgebraicArrays
using Unitful

import Base: *

Base.:*(a::Unitful.Units, b::VectorArray) = AlgebraicArray(a * vec(b), rangesize(b))
Base.:*(a::Unitful.Units, B::MatrixArray) = AlgebraicArray(a * Matrix(B), rangesize(B), domainsize(B))
Base.:*(B::Union{VectorArray,MatrixArray}, a::Unitful.Units) = a * B

end # module
