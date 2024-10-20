module AlgebraicArrays

using LinearAlgebra
using ArraysOfArrays

export VectorArray, MatrixArray, AlgebraicArray, Array
export VectorDimArray, MatrixDimArray
export parent, domaindims, rangedims, endomorphic
#export randn_VectorArray
export randn_MatrixArray
export # export Base methods
    size, show, vec, Matrix, *, first, Array
export # export more Base methods
    display, parent, \, /, real, exp
export # export more Base methods
    randn, fill, ones, zeros
export # export more Base methods
    getindex, setindex!, BroadcastStyle, similar
export # export LinearAlgebra methods
    transpose, adjoint, eigen, Diagonal
    
import Base: size, show, vec, Matrix, Array, axes
import Base: +, -, *, first, real , exp
import Base: display, parent, \, /, Array #, randn
import Base: getindex, setindex!, BroadcastStyle, similar
import Base: randn, fill, ones, zeros
import LinearAlgebra: transpose, adjoint, eigen, Diagonal

#struct VectorArray{T<:Number,N,A<:AbstractArray{T,N}} <: AbstractArray{T,1}
struct VectorArray{T,V <:AbstractVector{T}} <: AbstractVector{T}
    data:: V
    rangedims #::Union{Int,NTuple{N,Int}}
    #Union{Tuple,D1}, ddims::Union{Tuple,D2}) where T where D1 <: DimensionalData.Dimension where D2 <: DimensionalData.Dimension
end
VectorArray(A::AbstractArray) = VectorArray(vec(A),size(A)) 
VectorArray(a::Number) = a # helpful for slices that aren't vectors anymore
parent(b::VectorArray) = b.data
rangedims(b::VectorArray) = b.rangedims
#Base.size(b::VectorArray) = size(parent(b))
Base.size(b::VectorArray) = rangedims(b)
Base.vec(b::VectorArray) = parent(b)
#Base.axes(b::VectorArray) = axes(parent(b))

"""
    Array(A, rsize)

Construct an `Array` from a `VectorArray`

# Arguments
- `A::VectorArray`
- `rdims`: dimensions of range
"""
function Array(b::VectorArray)
    M = prod(rangedims(b))
    if M > 1
        return reshape(parent(b),rangedims(b))
    elseif M == 1
        # warning: introduces type instability
        # but useful for inner products
        return first(b)
    end
end

#Base.getindex(b::VectorArray, inds...) = getindex(parent(b), inds...)
Base.getindex(b::VectorArray, ind::Int) = VectorArray(parent(b)[ind]) # linear indexing 
#Base.getindex(b::VectorArray, ind::Int) = parent(b)[ind] # linear indexing 
# cartesian indexing
Base.getindex(b::VectorArray, inds::Vararg) = VectorArray(getindex(Array(b),inds...))
#Base.getindex(b::VectorArray, inds::Vararg) = getindex(Array(b),inds...)
#Base.getindex(b::VectorArray, inds::Vararg) = VectorArray(getindex(reshape(parent(b),rangedims(b)),inds...))
#Base.getindex(b::VectorArray, inds::Vararg) = b[LinearIndices(rangedims(b))[inds...]] # Stack Overflow 
#Base.getindex(b::VectorArray, inds::Vararg) = VectorArray([b[i] for i in LinearIndices(rangedims(b))[inds...]])

Base.setindex!(b::VectorArray, val, ind::Int) = parent(b)[ind] = val
Base.setindex!(b::VectorArray, val, inds::Vararg) = Array(b)[inds...] = val
#Base.setindex!(b::VectorArray, val, inds::Vararg) = reshape(parent(b),rangedims(b))[inds...] = val
        
# #Base.setindex!(b::VectorArray, v, inds...) = setindex!(parent(b), v, inds...) 
# function Base.setindex!(b::VectorArray, val, inds::Vararg)

#     for i in LinearIndices(rangedims(b))[inds...]
#         b.data[i] .= val
#     end
# end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorArray}}, ::Type{ElType}) where ElType
    # Scan the inputs
    A = find_va(bc)
    #println(axes(bc))
    VectorArray(similar(Array{ElType}, axes(bc)),A.rangedims)
end
Base.similar(b::VectorArray{T}) where T = VectorArray(similar(Array{T}, axes(b)),rangedims(b))

function Base.show(io::IO, mime::MIME"text/plain", b::VectorArray)
    #println(summary(b))
    show(io,mime,Array(b))
    println(io,"")
    println(io,"============================")
    println(io,"*operating algebraically as*")
    show(io,mime,vec(b))
end

# Base.IndexStyle(b::VectorArray) = Base.IndexStyle(Array(b))
# Base.iterate(b::VectorArray, args::Vararg) = iterate(Array(b), args...)
# # implement broadcast
Base.BroadcastStyle(::Type{<:VectorArray}) = Broadcast.ArrayStyle{VectorArray}()

# "`A = find_va(As)` returns the first VectorArray among the arguments."
find_va(bc::Base.Broadcast.Broadcasted) = find_va(bc.args)
find_va(args::Tuple) = find_va(find_va(args[1]), Base.tail(args))
find_va(x) = x
find_va(::Tuple{}) = nothing
find_va(a::VectorArray, rest) = a
find_va(::Any, rest) = find_va(rest)

end #module

# Base.transpose(P::VectorArray) = AlgebraicArray( transpose(vec(P)), 1, rangesize(P))

# #function Base.fill(val, rsize::Union{Int,NTuple{N,Int}}, type) where N
# function Base.fill(val, rsize, type) 
#     if type == :VectorArray
#         VectorArray(fill(val, rsize))
#     else
#         error("fill type not implemented")
#     end
# end

# #function Base.ones(rsize::Union{Int,NTuple{N,Int}}, type) where N
# function Base.ones(rsize, type) 
#     if type == :VectorArray
#         VectorArray(ones(rsize))
#     else
#         error("ones type not implemented")
#     end
# end

# #function Base.zeros(rsize::Union{Int,NTuple{N,Int}}, type) where N
# function Base.zeros(rsize, type)
#     if type == :VectorArray
#         VectorArray(zeros(rsize))
#     else
#         error("zeros type not implemented")
#     end
# end

# #function Base.randn(rsize::Union{Int,NTuple{N,Int}}, type) where N
# function Base.randn(rsize::Union{Int,NTuple{N,Int}},type::Symbol) where N
#     if type == :VectorArray
#         VectorArray(randn(rsize))
#     else
#         error("randn type not implemented")
#     end
# end


# #struct MatrixArray{T<:Number,
# struct MatrixArray{T,
#     M,
#     N,
#     R<:AbstractArray{T,M},
#     C<:AbstractArray{R,N}} <: AbstractArray{T,2}
#     data::C
# end

# """
#     AlgebraicArray(A,rsize,dsize)

# Construct a `VectoArray` or `MatrixArray` from an AbstractArray.

# # Arguments
# - `A::AbstractArray`
# - `rsize`: size of range
# - `dsize`: size of domain
# """
# function AlgebraicArray(A::AbstractMatrix{T},rsize::Union{Int,NTuple{N1,Int}},dsize::Union{Int,NTuple{N2,Int}}) where {N1,N2,T} # <: Number 

#     M = prod(dsize)
#     N = length(rsize)

#     if M > 1
#         P = Array{Array{T,N}}(undef,dsize)
#         for j in 1:M 
#             P[j] = reshape(A[:,j],rsize)
#         end
#         return MatrixArray(P)
#     elseif M == 1
#         # warning: introduces type instability
#         # but useful for transpose of row vector
#         return VectorArray(reshape(A,rsize))
#     else
#         error("incompatible number of columns") 
#     end
# end

# parent(A::MatrixArray) = A.data
# function Base.show(io::IO, mime::MIME"text/plain", A::MatrixArray)
#     show(io,mime,parent(A))
#     println(io,"")
#     println(io,"============================")
#     println(io,"*operating algebraically as*")
#     show(io,mime,Matrix(A))
# end
# Base.size(A::MatrixArray) = size(parent(A))
# Base.getindex(A::MatrixArray, inds::Vararg) = getindex(parent(A), inds...) # need to reverse order?
# Base.setindex!(A::MatrixArray, v, inds::Vararg) = setindex!(parent(A), v, inds...) # need to reverse order?
# domainsize(A::MatrixArray) = size(parent(A))
# rangesize(A::MatrixArray) = size(first(parent(A)))
# endomorphic(A::MatrixArray) = isequal(rangesize(A), domainsize(A))

# function Base.real(A::MatrixArray)

#     for j in eachindex(A)
#         A[j] = real.(A[j])
#     end
#     return A
#         #MatrixArray(real(parent(A)))
# end


# """
# function Matrix(P::MatrixArray{T}) where T <: Number
# """
# function Matrix(P::MatrixArray{T}) where T #<: Number
#     N = length(P) # number of columns/ outer dims
#     M = length(first(P)) # number of rows, take first inner element as example

#     A = Array{T}(undef,M,N)
#     if N > 1  
#         #for j in eachindex(P) # return Cartesian Index which fails on lhs
#         for j in 1:N # return Cartesian Index which fails on lhs
#             A[:,j] = P[j][:]
#         end
#     elseif N == 1
#         #for i in eachindex(first(P))
#         for i in 1:M #eachindex(first(P))
#             A[i,1] = first(P)[i] # keep it as a matrix
#         end
#     end
#     return A 
# end

# Array(P::MatrixArray) = Matrix(P)

# # a pattern for any function
# Base.transpose(P::MatrixArray) = AlgebraicArray( transpose(Matrix(P)), domainsize(P), rangesize(P))

# Base.adjoint(P::MatrixArray) = AlgebraicArray( adjoint(Matrix(P)), domainsize(P), rangesize(P))

# # function Base.:*(A::MatrixArray, b::VectorArray)
# #     c = zero(first(A))
# #     for j in eachindex(A)
# #         c += A[j] * b[j]
# #     end
# #     return VectorArray(c)
# # end

# # slightly faster version in a one-liner form
# Base.:*(A::MatrixArray, b::VectorArray) =  AlgebraicArray(Matrix(A) * vec(b), rangesize(A))
# Base.:*(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) * Matrix(B), rangesize(A), domainsize(B))
# Base.:*(a::VectorArray, B::MatrixArray) = AlgebraicArray(vec(a) * Matrix(B), rangesize(a), domainsize(B))
# Base.:*(a::Number, B::MatrixArray) = AlgebraicArray(a * Matrix(B), rangesize(B), domainsize(B))
# Base.:*(B::MatrixArray, a::Number) = a * B

# Base.:(\ )(A::MatrixArray, b::VectorArray) = AlgebraicArray(Matrix(A) \ vec(b), domainsize(A))
# Base.:(\ )(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) \ Matrix(B), domainsize(A), domainsize(B))
# #     (c isa Number) && (c = [c]) # useful snippet if one-linear fails in some cases

# Base.:+(A::MatrixArray, B::MatrixArray) = MatrixArray(parent(A) + parent(B))
# Base.:+(a::VectorArray, b::VectorArray) = VectorArray(parent(a) + parent(b))

# Base.:-(A::MatrixArray, B::MatrixArray) = MatrixArray(parent(A) - parent(B))
# Base.:-(a::VectorArray, b::VectorArray) = VectorArray(parent(a) - parent(b))

# """
# function matrix right divide

# `A/B = ( B'\\A')'
# """
# Base.:(/)(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) / Matrix(B), rangesize(A), rangesize(B))

# # function randn_MatrixArray(rsize::Union{Int,NTuple{N1,Int}},dsize::Union{Int,NTuple{N2,Int}}) where {N1,N2}
# #     # make an array of arrays
# #     alldims = Tuple(vcat([i for i in rsize],[j for j in dsize]))
# #     # warning, doesn't work for 3D+ arrays
# #     return MatrixArray(Matrix(nestedview(randn(alldims),length(dsize))))
# # end
# function randn(rsize::Union{Int,NTuple{N1,Int}},dsize::Union{Int,NTuple{N2,Int}},type::Symbol) where {N1,N2}
#     if type == :MatrixArray
#         # make an array of arrays
#         alldims = Tuple(vcat([i for i in rsize],[j for j in dsize]))
#         # warning, doesn't work for 3D+ arrays
#         return MatrixArray(Matrix(nestedview(randn(alldims),length(dsize))))
#     else
#         error("randn not implemented for this type")
#     end
# end


# function LinearAlgebra.eigen(A::MatrixArray)
#     F = eigen(Matrix(A))
#     dsize = length(F.values)
#     rsize = rangesize(A)
#     values = AlgebraicArray(F.values,dsize)
#     vectors = AlgebraicArray(F.vectors,rsize,dsize) 
#     return Eigen(values, vectors)
# end

# Diagonal(a::VectorArray) = AlgebraicArray(Diagonal(vec(a)), rangesize(a), rangesize(a))

# function exp(A::MatrixArray)
#     # A must be endomorphic (check type signature someday)
#     !endomorphic(A) && error("A must be endomorphic to be consistent with matrix exponential")
#     eA = exp(Matrix(A)) # move upstream to MultipliableDimArrays eventually
#     return AlgebraicArray(exp(Matrix(A)),rangesize(A),domainsize(A)) # wrap with same labels and format as A
# end

# end # module AlgebraicArrays
