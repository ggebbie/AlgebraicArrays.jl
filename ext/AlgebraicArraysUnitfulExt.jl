module AlgebraicArraysUnitfulExt

using AlgebraicArrays
using LinearAlgebra
using Unitful

import Base: *, (\), (/)
import Unitful: ustrip, unit
import LinearAlgebra: eigen

# Base.:*(a::Unitful.Units, b::VectorArray) = AlgebraicArray(a * vec(b), rangedims(b))
# Base.:*(a::Unitful.Units, B::MatrixArray) = AlgebraicArray(a * Matrix(B), rangedims(B), domaindims(B))
# Base.:*(B::Union{VectorArray,MatrixArray}, a::Unitful.Units) = a * B

# Unitful doesn't handle matrix left divide between Quantity and non-Quantity
# nor this case.
# this is a benign form of type piracy
function Base.:(\)(A::Union{AbstractVecOrMat{Quantity{Q1,S1,V1}},Diagonal{Quantity{Q1,S1,V1}}},
                   B::AbstractVecOrMat{Quantity{Q2,S2,V2}}) where {Q1,S1,V1} where {Q2,S2,V2}
     Aunit = unit(first(A))
     Bunit = unit(first(B))
     return (Bunit/Aunit) * (ustrip.(A) \ ustrip.(B))
end

# # more type piracy
# function Base.:(\)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat) where {Q1,S1,V1} 
#     #if uniform(A) # already handled by input types
#     Aunit = unit(first(first(A)))
#     return (1/Aunit) * (ustrip.(A) \ B)
# end

# Unitful is not handling this case now, benign type piracy here
function Base.:(/)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat) where {Q1,S1,V1} 
    Aunit = unit(first(A))
    return Aunit * (ustrip.(A) / B)
end

function Base.:(/)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat{Quantity{Q2,S2,V2}}) where {Q1,S1,V1} where {Q2,S2,V2} 
     Aunit = unit(first(first(A)))
     Bunit = unit(first(first(B)))
    return (Aunit/Bunit) * (ustrip.(A) / ustrip.(B))
 end

Base.:(/)(A::MatrixArray, b::Unitful.Units) = AlgebraicArray(Matrix(A)/b, rangedims(A), domaindims(A))

function Base.:(/)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}},
    B::AbstractVecOrMat{Quantity{Q2,S2,V2}}) where {Q1,S1,V1} where {Q2,S2,V2}
    C = AlgebraicArrays.matrix_or_vec(A) / AlgebraicArrays.matrix_or_vec(B)
    return AlgebraicArray(C, (rangedims(A), domaindims(A)))
end

# # caution: dot broadcast added here on rhs, not lhs
# Unitful.ustrip(A::MatrixArray) = AlgebraicArray(ustrip.(Matrix(A)), rangedims(A), domaindims(A))

# function LinearAlgebra.eigen(A::MatrixArray{T,D,N,Matrix{Quantity{T2,S,V}}}) where {T1,T2,N,M,S,V}
function LinearAlgebra.eigen(A::MatrixArray{<:Quantity})
    !uniform(A) && error("A has heterogeneous units, no eigenstructure")
    F = eigen(Matrix(A)) 
    dsize = size(F.values)
    rsize = rangedims(A)
    values = AlgebraicArray(F.values,(dsize,))
    vectors = AlgebraicArray(F.vectors,(rsize,dsize)) 
    return Eigen(values, vectors)
end

# Unitful doesn't cover simple unitful eigenvalues either
# this type signature should be (is?) restricted to uniform matrices
function LinearAlgebra.eigen(A::AbstractMatrix{Quantity{T,S,V}}) where {T,S,V}
    Aunit = unit(first(A))
    F = eigen(ustrip.(A))
    return Eigen(F.values*Aunit, F.vectors)
end

# #AbstractMatrix(F::Eigen) = F.vectors * Diagonal(F.values) / F.vectors

uniform(A::AbstractArray{<:Quantity{D,E}}) where {D,E} = true
uniform(A::AbstractArray) = false
uniform(A::AlgebraicArray) = uniform(parent(A))

end # module
