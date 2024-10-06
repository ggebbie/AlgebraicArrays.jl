module AlgebraicArrays

using LinearAlgebra

export VectorArray, MatrixArray
export parent, domainsize, rangesize 
#export CRmult
export # export Base methods
    size, show, vec, Matrix, *, first,  display, parent, \
export # export LinearAlgebra methods
    transpose, adjoint
    
import Base: size, show, vec, Matrix, *, first, display, parent, \
import LinearAlgebra: transpose, adjoint

struct VectorArray{T<:Number,N,A<:AbstractArray{T,N}} <: AbstractArray{T,1}
    data:: A
end

"""
    VectorArray(A, rsize)

Construct a `VectorArray` from an AbstractArray.

# Arguments
- `A::AbstractArray`
- `rsize`: size of range
"""
VectorArray(A::AbstractVector, rsize) = VectorArray(reshape(A,rsize))

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

struct MatrixArray{T<:Number,
    M,
    N,
    R<:AbstractArray{T,M},
    C<:AbstractArray{R,N}} <: AbstractArray{T,2}
    data::C
end

"""
    MatrixArray(A,rsize,dsize)

Construct a `MatrixArray` from an AbstractArray.

# Arguments
- `A::AbstractArray`
- `rsize`: size of range
- `dsize`: size of domain
"""
function MatrixArray(A::AbstractMatrix{T},rsize,dsize) where T <: Number 

    M = prod(dsize)
    N = length(rsize)
    P = Array{Array{T,N}}(undef,dsize)
    for j in 1:M 
        P[j] = reshape(A[:,j],rsize)
    end
    return MatrixArray(P)
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
function Matrix(P::MatrixArray{T}) where T <: Number
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

# a pattern for any function
Base.transpose(P::MatrixArray) = MatrixArray( transpose(Matrix(P)), domainsize(P), rangesize(P))

Base.adjoint(P::MatrixArray) = MatrixArray( adjoint(Matrix(P)), domainsize(P), rangesize(P))

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

# slightly faster as a one-liner
Base.:*(A::MatrixArray, b::VectorArray) =  VectorArray(Matrix(A) * vec(b), rangesize(A))

Base.:*(A::MatrixArray, B::MatrixArray) = MatrixArray(Matrix(A) * Matrix(B), rangesize(A), domainsize(B))

# function Base.:*(A::MatrixArray, B::MatrixArray) 
#     C = Matrix(A) * Matrix(B)
#     (C isa Number) && (C = [C])

#     #reshape using all of the dimensions
#     #rsize = size(first(A))
#     #dsize  = size(B)
#     return MatrixArray(C, rangesize(A), domainsize(B))
#     #return MatrixArray(C,rsize,dsize)
# end

Base.:(\ )(A::MatrixArray, b::VectorArray) = VectorArray(Matrix(A) \ vec(b), domainsize(A))
Base.:(\ )(A::MatrixArray, B::MatrixArray) = MatrixArray(Matrix(A) \ Matrix(B), domainsize(A), domainsize(B))

# function Base.:(\)(A::MatrixArray, b::VectorArray) #where T1<: AbstractDimArray where T2 <: Number
#     c = Matrix(A) \ vec(b)
#     (c isa Number) && (c = [c])
#     return VectorArray(c, rangesize(A))
# end

# function Base.:(\)(A::DimArray{T1}, B::DimArray{T2})  where T1 <: AbstractDimArray where T2 <: AbstractDimArray 
#     C = Matrix(A) \ Matrix(B)
#     (C isa Number) && (C = [C])
#     return MatrixArray(C, rangesize(A), rangesize(B))
# end

end # module AlgebraicArrays
