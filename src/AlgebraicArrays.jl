module AlgebraicArrays

using LinearAlgebra
using ArraysOfArrays

export VectorArray, MatrixArray, AlgebraicArray, Array
export parent, domainsize, rangesize
export randn_VectorArray, randn_MatrixArray
export # export Base methods
    size, show, vec, Matrix, *, first
export # export more Base methods
    display, parent, \, / #, randn
export # export LinearAlgebra methods
    transpose, adjoint, eigen, Diagonal
    
import Base: size, show, vec, Matrix, *, first
import Base: display, parent, \, /, Array #, randn 
import LinearAlgebra: transpose, adjoint, eigen, Diagonal

struct VectorArray{T<:Number,N,A<:AbstractArray{T,N}} <: AbstractArray{T,1}
    data:: A
end

"""
    AlgebraicArray(A, rsize)

Construct a `VectorArray` or `MatrixArray` from an AbstractArray.

# Arguments
- `A::AbstractArray`
- `rsize`: size of range
"""
#VectorArray(A::AbstractVector, rsize) = VectorArray(reshape(A,rsize))
function AlgebraicArray(A::AbstractVector, rsize::Union{Int,NTuple{N,Int}}) where N
    M = prod(rsize)
    if M > 1
        return VectorArray(reshape(A,rsize))
    elseif M == 1
        # warning: introduces type instability
        # but useful for inner products
        return first(A)
    end
end
         
parent(b::VectorArray) = b.data
function Base.display(b::VectorArray)
    #println(summary(b))
    display(parent(b))
    println("operating algebraically as")
    display(vec(b))
end
#Base.display(b::VectorArray) = display(parent(b))
Base.show(b::VectorArray) = show(parent(b))
Base.size(b::VectorArray) = size(parent(b))
Base.vec(b::VectorArray) = vec(parent(b))
Base.getindex(b::VectorArray, inds...) = getindex(parent(b), inds...)

rangesize(b::VectorArray) = size(parent(b))
domainsize(b::VectorArray) = ()

Base.transpose(P::VectorArray) = AlgebraicArray( transpose(vec(P)), 1, rangesize(P))

randn_VectorArray(rsize) = VectorArray(randn(rsize))

struct MatrixArray{T<:Number,
    M,
    N,
    R<:AbstractArray{T,M},
    C<:AbstractArray{R,N}} <: AbstractArray{T,2}
    data::C
end

"""
    AlgebraicArray(A,rsize,dsize)

Construct a `VectoArray` or `MatrixArray` from an AbstractArray.

# Arguments
- `A::AbstractArray`
- `rsize`: size of range
- `dsize`: size of domain
"""
function AlgebraicArray(A::AbstractMatrix{T},rsize::Union{Int,NTuple{N1,Int}},dsize::Union{Int,NTuple{N2,Int}}) where {N1,N2,T} # <: Number 

    M = prod(dsize)
    N = length(rsize)

    if M > 1
        P = Array{Array{T,N}}(undef,dsize)
        for j in 1:M 
            P[j] = reshape(A[:,j],rsize)
        end
        return MatrixArray(P)
    elseif M == 1
        # warning: introduces type instability
        # but useful for transpose of row vector
        return VectorArray(reshape(A,rsize))
    else
        error("incompatible number of columns") 
    end
end

parent(A::MatrixArray) = A.data
Base.show(A::MatrixArray) = show(parent(A))
function Base.display(A::MatrixArray)
    #println(summary(b))
    display(parent(A))
    println("operating algebraically as")
    display(Matrix(A))
end
#Base.display(A::MatrixArray) = display(parent(A))
#Base.size(A::MatrixArray) = size(Matrix(A))
Base.size(A::MatrixArray) = size(parent(A))
Base.getindex(A::MatrixArray, inds...) = getindex(parent(A), inds...) # need to reverse order?

domainsize(A::MatrixArray) = size(parent(A))
rangesize(A::MatrixArray) = size(first(parent(A)))

"""
function Matrix(P::MatrixArray{T}) where T <: Number
"""
function Matrix(P::MatrixArray{T}) where T #<: Number
    N = length(P) # number of columns/ outer dims
    M = length(first(P)) # number of rows, take first inner element as example

    A = Array{T}(undef,M,N)
    if N > 1  
        #for j in eachindex(P) # return Cartesian Index which fails on lhs
        for j in 1:N # return Cartesian Index which fails on lhs
            A[:,j] = P[j][:]
        end
    elseif N == 1
        #for i in eachindex(first(P))
        for i in 1:M #eachindex(first(P))
            A[i,1] = first(P)[i] # keep it as a matrix
        end
    end
    return A 
end

Array(P::MatrixArray) = Matrix(P)

# a pattern for any function
Base.transpose(P::MatrixArray) = AlgebraicArray( transpose(Matrix(P)), domainsize(P), rangesize(P))

Base.adjoint(P::MatrixArray) = AlgebraicArray( adjoint(Matrix(P)), domainsize(P), rangesize(P))

# function Base.:*(A::MatrixArray, b::VectorArray)
#     c = zero(first(A))
#     for j in eachindex(A)
#         c += A[j] * b[j]
#     end
#     return VectorArray(c)
# end
#slightly faster version of multiplication
# function Base.:*(A::MatrixArray, b::VectorArray)
#     C = Matrix(A) * vec(b)
#     (C isa Number) && (C = [C])
#     rowdims = size(first(A))
#     return VectorArray(reshape(C,rowdims))
# end

# would prefer rand(MatrixArray,rsize,dsize)
function rand_MatrixArray(rsize,dsize)
    # make an array of arrays
    alldims = Tuple(vcat([i for i in rsize],[j for j in dsize]))
    return MatrixArray(Matrix(nestedview(randn(alldims),length(dsize))))
end

# slightly faster as a one-liner
Base.:*(A::MatrixArray, b::VectorArray) =  AlgebraicArray(Matrix(A) * vec(b), rangesize(A))

Base.:*(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) * Matrix(B), rangesize(A), domainsize(B))

Base.:*(a::VectorArray, B::MatrixArray) = AlgebraicArray(vec(a) * Matrix(B), rangesize(a), domainsize(B))

# function Base.:*(A::MatrixArray, B::MatrixArray) 
#     C = Matrix(A) * Matrix(B)
#     (C isa Number) && (C = [C])

#     #reshape using all of the dimensions
#     #rsize = size(first(A))
#     #dsize  = size(B)
#     return MatrixArray(C, rangesize(A), domainsize(B))
#     #return MatrixArray(C,rsize,dsize)
# end

Base.:(\ )(A::MatrixArray, b::VectorArray) = AlgebraicArray(Matrix(A) \ vec(b), domainsize(A))
Base.:(\ )(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) \ Matrix(B), domainsize(A), domainsize(B))
#     (c isa Number) && (c = [c]) # useful snippet if one-linear fails in some cases

"""
function matrix right divide

`A/B = ( B'\\A')'
"""
# function Base.:(/)(A::DimArray{T1}, B::DimArray{T2})  where T1 <: AbstractDimArray where T2 <: AbstractDimArray 
#     Amat = Matrix(A) / Matrix(B)
#     (Amat isa Number) && (Amat = [Amat])
#     ddims = dims(first(B))
#     rdims = dims(first(A))
#     return MultipliableDimArray(Amat, rdims, ddims)
# end
Base.:(/)(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) / Matrix(B), rangesize(A), rangesize(B))

function randn_MatrixArray(rsize::Union{Int,NTuple{N1,Int}},dsize::Union{Int,NTuple{N2,Int}}) where {N1,N2}
    # make an array of arrays
    alldims = Tuple(vcat([i for i in rsize],[j for j in dsize]))
    return MatrixArray(Matrix(nestedview(randn(alldims),length(dsize))))
end

# eigenstructure only exists if A is uniform
# should be a better way by reading type
# uniform(A::MatrixArray{Real}) = true
# uniform(b::VectorArray{Real}) = true
# uniform(A) = uniform(Matrix(A))
# function uniform(A::Matrix)
#     ulist = unit.(A)
#     return allequal(ulist)
# end

function LinearAlgebra.eigen(A::MatrixArray)
    F = eigen(Matrix(A))
    dsize = length(F.values)
    rsize = rangesize(A)
    values = AlgebraicArray(F.values,dsize)
    vectors = AlgebraicArray(F.vectors,rsize,dsize) 
    return Eigen(values, vectors)
end

Diagonal(a::VectorArray) = AlgebraicArray(Diagonal(vec(a)), rangesize(a), rangesize(a))

end # module AlgebraicArrays
