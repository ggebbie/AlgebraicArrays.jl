module AA_DD_UnitfulExt

using AlgebraicArrays
using DimensionalData
using DimensionalData:@dim
using Unitful

using LinearAlgebra

@dim Eigenmode "eigenmode"

# export VectorDimArray, MatrixDimArray, AlgebraicDimArray
# export dims, rowvector, AlgebraicArray
# export rand, randn, zeros, ones

# import AlgebraicArrays: rangedims, domaindims, AlgebraicArray
# import AlgebraicArrays: MatrixArray, VectorArray 
import LinearAlgebra: eigen
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

function LinearAlgebra.eigen(A::MatrixDimArray{<:Quantity})
    !endomorphic(A) && error("AlgebraicArrays.jl: not endomorphic")
    F = eigen(Matrix(A))

    eigen_dims = Eigenmode(1:length(F.values))
    newdim = (rangedims(A)..., eigen_dims)
    varr = reshape(F.vectors, size(newdim)...)
    vda = DimArray(varr, newdim)
    rdims_new = size(rangedims(A))
    ddims_new = size(eigen_dims)

    vectors = AlgebraicArray(vda,(rdims_new,ddims_new))

    # arr = AlgebraicArray(F.values, (size(eigen_dims),))
    arr = AlgebraicArray(F.values, (ddims_new,))
    da = DimArray(arr, eigen_dims)
    values = AlgebraicArray(da, (size(eigen_dims),))

    return Eigen(values, vectors)
end

end # module
