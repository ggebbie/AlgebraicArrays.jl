module AlgebraicArraysUnitfulExt

using AlgebraicArrays
using LinearAlgebra
using Unitful

import Base: *, (\)
import Unitful: ustrip, unit
import LinearAlgebra: eigen

Base.:*(a::Unitful.Units, b::VectorArray) = AlgebraicArray(a * vec(b), rangedims(b))
Base.:*(a::Unitful.Units, B::MatrixArray) = AlgebraicArray(a * Matrix(B), rangedims(B), domaindims(B))
Base.:*(B::Union{VectorArray,MatrixArray}, a::Unitful.Units) = a * B

# Unitful doesn't handle matrix left divide between Quantity and non-Quantity
# nor this case.
# this is a benign form of type piracy
function Base.:(\)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat{Quantity{Q2,S2,V2}}) where {Q1,S1,V1} where {Q2,S2,V2}
    #if uniform(A) # already handled by input types
    Aunit = unit(first(first(A)))
    Bunit = unit(first(first(B)))
    return (Bunit/Aunit) * (ustrip.(A) \ ustrip.(B))
end

# more type piracy
function Base.:(\)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat) where {Q1,S1,V1} 
    #if uniform(A) # already handled by input types
    Aunit = unit(first(first(A)))
    return (1/Aunit) * (ustrip.(A) \ B)
end

# handle issue 21
function Base.:(\)(A::LinearAlgebra.Diagonal{Unitful.Quantity{Q1, S1, V1}, V} where V<:AbstractArray{Unitful.Quantity{Q1, S1, V1}, 1}, B::AbstractArray{Unitful.Quantity{Q2, S2, V2}, 2}) where {Q2, S2, V2, Q1, S1, V1}
#function Base.:(\)(A::Diagonal{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat) where {Q1,S1,V1} 
    #if uniform(A) # already handled by input types
    Aunit = unit(first(first(A)))
    return (1/Aunit) * (ustrip.(A) \ B)
end

# Unitful not handling this case, benign type piracy here
function Base.:(/)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat) where {Q1,S1,V1} 
    Aunit = unit(first(first(A)))
    return Aunit * (ustrip.(A) / B)
end

function Base.:(/)(A::AbstractVecOrMat{Quantity{Q1,S1,V1}}, B::AbstractVecOrMat{Quantity{Q2,S2,V2}}) where {Q1,S1,V1} where {Q2,S2,V2} 
    Aunit = unit(first(first(A)))
    Bunit = unit(first(first(B)))
    return (Aunit/Bunit) * (ustrip.(A) / ustrip.(B))
end
Base.:(/)(A::MatrixArray, b::Unitful.Units) = AlgebraicArray(Matrix(A)/b, rangedims(A), domaindims(A))

# caution: dot broadcast added here on rhs, not lhs
Unitful.ustrip(A::MatrixArray) = AlgebraicArray(ustrip.(Matrix(A)), rangedims(A), domaindims(A))

function LinearAlgebra.eigen(A::MatrixArray{T1,N,M,Matrix{Quantity{T2,S,V}}}) where {T1,T2,N,M,S,V}

    Aunit = unit(first(first(A)))
    F = eigen(ustrip.(Matrix(A)))
    dsize = length(F.values)
    rsize = rangedims(A)
    values = AlgebraicArray(F.values*Aunit,dsize)
    vectors = AlgebraicArray(F.vectors,rsize,dsize) 
    return Eigen(values, vectors)
end

# Unitful doesn't cover simple unitful eigenvalues either
# this type signature should be restricted to uniform matrices
function LinearAlgebra.eigen(A::AbstractMatrix{Quantity{T,S,V}}) where {T,S,V}
    Aunit = unit(first(A))
    F = eigen(ustrip.(A))
    return Eigen(F.values*Aunit, F.vectors)
end

#AbstractMatrix(F::Eigen) = F.vectors * Diagonal(F.values) / F.vectors

end # module
