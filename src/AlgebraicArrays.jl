module AlgebraicArrays

using LinearAlgebra

export VectorArray, MatrixArray, AlgebraicArray, Array
export VectorDimArray, MatrixDimArray
export parent, domaindims, rangedims, endomorphic, rowvector
export # export Base methods
    size, show, vec, Matrix, *, first
export # export more Base methods
    display, parent, \, /, real, exp
export # export more Base methods
    rand, randn, fill, ones, zeros
export # export more Base methods
    getindex, setindex!, BroadcastStyle, similar
export # export more Base methods
    IndexStyle, eachindex, iterate 
export # export LinearAlgebra methods
    transpose, adjoint, eigen, Diagonal, diag

export AArray

import Base: size, show, vec, Matrix
import Base: +, -, *, first, real , exp
import Base: display, parent, \, /, Array #, randn
import Base: getindex, setindex!, BroadcastStyle, similar
import Base: rand, randn, fill, ones, zeros
import LinearAlgebra: transpose, adjoint, eigen, Diagonal, diag

# T: numeric type
# N: number of total dimensions (sum of rows and columns)
# D: dimension=1 for VectorArray, 2 for MatrixArray
struct AlgebraicArray{T,D,N} <: AbstractArray{T,D}
    data:: AbstractArray{T,N}
    dims:: NTuple{D,Tuple}
    function AlgebraicArray(x::AbstractArray{T,N},y::NTuple{D,Tuple}) where {T,D,N}
        return new{T,D,N}(x,y)
    end
end

AlgebraicArray(a::Number, b) = a # helpful for slices that aren't vectors anymore

# size of data array
asize(A::AlgebraicArray) = unwrap(A.dims)
unwrap(d::NTuple{D,Tuple{I,I}}) where D where I <: Int =  Tuple(d[i][j] for i in eachindex(d) for j in eachindex(d[i]))

parent(A::AlgebraicArray) = A.data
Base.size(A::AlgebraicArray) = prod.(A.dims)

# subset of all AArrays is a VArray (VectorArray)
VectorArray{T,N} = AlgebraicArray{T,1,N}

function Base.getindex(b::VectorArray, inds::Vararg)
    tmp =  getindex(b.data, inds...)
    newsize = (size(tmp),)
    return AlgebraicArray( tmp, (size(tmp),))
end

# Base.getindex(b::VectorArray, inds::Vararg) = VectorArray(getindex(parent(b), inds...))
# Base.getindex(b::VectorArray; kw...) = VectorArray(getindex(parent(b); kw...))
# Base.getindex(b::VectorArray; kw...) = getindex(parent(b); kw...)
Base.getindex(b::VectorArray; kw...) = getindex(vec(b); kw...)
# Base.getindex(b::VectorArray; ind::Tuple) = getindex(parent(b), ind...)

Base.vec(b::VectorArray) = vec(b.data)

function Base.show(io::IO, mime::MIME"text/plain", b::VectorArray)
    #println(summary(b))
    show(io,mime,parent(b))
    println(io,"")
    println(io,"============================")
    println(io,"*operating algebraically as*")
    show(io,mime,vec(b))
end

# #Base.getindex(b::VectorArray, inds...) = getindex(parent(b), inds...)
# #Base.getindex(A::VectorArray, inds::Vararg) = VectorArray(A.data[inds...])
# #Base.dotview(b::VectorArray, inds::Vararg) = VectorArray(dotview(parent(b); inds...))
# #Base.dotview(b::VectorArray, inds::Vararg; kw...) = VectorArray(DimensionalData.dotview(parent(b), inds..., kw...))

# # function Base.getindex(b::VectorArray, inds...)
# #     #I = to_indices(parent(parent(b)), (inds...))
# #     tmp = getindex(parent(b), inds...)

# #     # check for any slices
# #     if (length(tmp) > 1) && !isa(tmp, VectorArray)
# #         return VectorArray(tmp)
# #     else
# #         return tmp
# #     end
# #end
#     #     @eval @propagate_inbounds function Base.$f(A::AbstractDimArray, i1::StandardIndices, i2::StandardIndices, Is::StandardIndices...)
#     #         I = to_indices(A, (i1, i2, Is...))
#     #         x = Base.$f(parent(A), I...)
#     #         all(i -> i isa Integer, I) ? x : rebuildsliced(Base.$f, A, x, I)
#     #     end
#     # end

Base.setindex!(b::VectorArray, val, inds::Vararg) = b.data[inds...] = val
# Base.setindex!(b::VectorArray, v, inds...) = setindex!(parent(b), v, inds...) 
# Base.setindex!(b::VectorArray, v; kw...) = setindex!(parent(b), v, kw...) 
# Base.iterate(b::VectorArray, args::Vararg) = iterate(parent(b), args...)

# # `VectorArray` is a subtype of AbstractVector which causes issues with eachindex
# # What other fundamental operators need adjustment?
# Base.eachindex(b::VectorArray) = eachindex(parent(b))

# Base.IndexStyle(b::VectorArray) = Base.IndexStyle(parent(b))
# Base.axes(b::VectorArray,d) = axes(parent(b),d)
# rangedims(b::VectorArray) = size(parent(b))
# domaindims(b::VectorArray) = ()
# #Base.real(b::VectorArray) = VectorArray(real(parent(b)))
# Base.transpose(P::VectorArray) = AlgebraicArray( transpose(vec(P)), 1, rangedims(P))


# #Base.dotview(b::VectorArray, inds::Vararg) = VectorArray(dotview(parent(b); inds...))
# #Base.dotview(b::VectorArray, inds::Vararg; kw...) = VectorArray(DimensionalData.dotview(parent(b), inds..., kw...))

# # function Base.getindex(b::VectorArray, inds...)
# #     #I = to_indices(parent(parent(b)), (inds...))
# #     tmp = getindex(parent(b), inds...)

# #     # check for any slices
# #     if (length(tmp) > 1) && !isa(tmp, VectorArray)
# #         return VectorArray(tmp)
# #     else
# #         return tmp
# #     end
# #end
#     #     @eval @propagate_inbounds function Base.$f(A::AbstractDimArray, i1::StandardIndices, i2::StandardIndices, Is::StandardIndices...)
#     #         I = to_indices(A, (i1, i2, Is...))
#     #         x = Base.$f(parent(A), I...)
#     #         all(i -> i isa Integer, I) ? x : rebuildsliced(Base.$f, A, x, I)
#     #     end
#     # end

# #Base.setindex!(b::VectorArray, val, inds::Vararg) = b.data[inds...] = val
# Base.setindex!(b::VectorArray, v, inds...) = setindex!(parent(b), v, inds...) 
# Base.setindex!(b::VectorArray, v; kw...) = setindex!(parent(b), v, kw...) 
# Base.iterate(b::VectorArray, args::Vararg) = iterate(parent(b), args...)

# # `VectorArray` is a subtype of AbstractVector which causes issues with eachindex
# # What other fundamental operators need adjustment?
# Base.eachindex(b::VectorArray) = eachindex(parent(b))

# Base.IndexStyle(b::VectorArray) = Base.IndexStyle(parent(b))
# Base.axes(b::VectorArray,d) = axes(parent(b),d)
# rangedims(b::VectorArray) = size(parent(b))
# domaindims(b::VectorArray) = ()
# #Base.real(b::VectorArray) = VectorArray(real(parent(b)))
# Base.transpose(P::VectorArray) = AlgebraicArray( transpose(vec(P)), 1, rangedims(P))

#function Base.fill(val, rsize::Union{Int,NTuple{N,Int}}, type) where N
Base.fill(val::T, dims::NTuple{D,Tuple}) where {T,D} =
    AlgebraicArray(fill(val, AlgebraicArrays.unwrap(dims)), dims)

Base.zeros(T::Type, dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( zeros(T, AlgebraicArrays.unwrap(dims)), dims)

Base.zeros(dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( zeros(AlgebraicArrays.unwrap(dims)), dims)

Base.ones(T::Type, dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( ones(T, AlgebraicArrays.unwrap(dims)), dims)

Base.ones(dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( ones(AlgebraicArrays.unwrap(dims)), dims)

Base.rand(T::Type, dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( rand(T, AlgebraicArrays.unwrap(dims)...), dims)

Base.rand(dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( rand(AlgebraicArrays.unwrap(dims)...), dims)

Base.randn(T::Type, dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( randn(T, AlgebraicArrays.unwrap(dims)), dims)

Base.randn(dims::NTuple{D,Tuple}) where D =
    AlgebraicArray( randn(AlgebraicArrays.unwrap(dims)), dims)

# # implement broadcast
Base.BroadcastStyle(::Type{<:VectorArray}) = Broadcast.ArrayStyle{VectorArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorArray}}, ::Type{ElType}) where ElType
    # Scan the inputs
    A = find_va(bc)
    AlgebraicArray(similar(Array{ElType}, axes(bc)), bc.args)
end
function Base.similar(va::VectorArray{T}) where T 
    tmp = reshape(similar(Array{T}, axes(va)), 
        AlgebraicArrays.unwrap(va.dims))
    
    return AlgebraicArray(tmp, va.dims)
    # VectorArray(similar(Array{T}, axes(va)))
end

"`A = find_va(As)` returns the first VectorArray among the arguments."
find_va(bc::Base.Broadcast.Broadcasted) = find_va(bc.args)
find_va(args::Tuple) = find_va(find_va(args[1]), Base.tail(args))
find_va(x) = x
find_va(::Tuple{}) = nothing
find_va(a::VectorArray, rest) = a
find_va(::Any, rest) = find_va(rest)

# subset of all AArrays is a VArray (VectorArray)
MatrixArray{T,N} = AlgebraicArray{T,2,N}

# example with two indices
function Base.getindex(A::MatrixArray, rowind::Int, colind::Int)
    # reshape A to a Matrix
    A2 = reshape(A.data, size(A))
    return getindex(A2, rowind, colind)
end


# #struct MatrixArray{T<:Number,
# struct MatrixArray{T,
#     M,
#     N,
#     R<:AbstractArray{T,M},
#     C<:AbstractArray{R,N}} <: AbstractArray{T,2}
#     data::C
# end

# # sometimes a singleton matrix is needed
# # force it to happen with this constructor
# function MatrixArray(A::AbstractMatrix{T},rsize::Union{Int,NTuple{N1,Int}},dsize::Union{Int,NTuple{N2,Int}}) where {N1,N2,T} # <: Number 

#     M = prod(dsize)
#     N = length(rsize)
#     P = Array{Array{T,N}}(undef,dsize)
#     for j in 1:M 
#         P[j] = reshape(A[:,j],rsize)
#     end
#     return MatrixArray(P)
# end

# # unknown whether `B` is a VectorArray or MatrixArray.
# AlgebraicArray(B::C) where {T,M,N,R<:AbstractArray{T,M},C<:AbstractArray{R,N}} = MatrixArray(B)
# # looks like a matrix, but only has one column
# #AlgebraicArray(B::C) where {T,M,R<:AbstractArray{T,M},C<:AbstractArray{R,1}} = VectorArray(first(B))

# """
#     AlgebraicArray(A,rsize,dsize)

# Construct a `VectorArray` or `MatrixArray` from an AbstractArray.

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
function Base.show(io::IO, mime::MIME"text/plain", A::MatrixArray)
    show(io,mime,parent(A))
    println(io,"")
    println(io,"============================")
    println(io,"*operating algebraically as*")
    show(io,mime,Matrix(A))
end
# Base.size(A::MatrixArray) = size(parent(A))
Matrix(P::MatrixArray) = reshape( P.data, size(P))

# function Base.getindex(A::MatrixArray, inds::Vararg)
#     Aslice = getindex(parent(A), inds...)
#     return AlgebraicArray(Aslice)
# end

# rowvector(A::MatrixArray, rowindex::Vararg) = transpose(VectorArray([A[j][rowindex...] for j in eachindex(A)]))
    
# Base.getindex(A::MatrixArray; kw...) = getindex(parent(A), kw...) 
# Base.setindex!(A::MatrixArray, v, inds::Vararg) = setindex!(parent(A), v, inds...) # need to reverse order?
# Base.setindex!(A::MatrixArray, v; kw...) = setindex!(parent(A), v, kw...) 
# #Base.IndexStyle(A::MatrixArray) = Base.IndexStyle(parent(A))

domaindims(A::MatrixArray) = first(A.dims)
rangedims(A::MatrixArray) = last(A.dims)
endomorphic(A::MatrixArray) = isequal(rangedims(A), domaindims(A))

function LinearAlgebra.diag(A::MatrixArray)
    if endomorphic(A)
        return AlgebraicArray(diag(Matrix(A)),(rangedims(A),))
    else
        # unclear what to do about dimensions in this case
        # punt and return a vector, warning: type unstable
        return diag(Matrix(A))
    end
end 

# function Base.real(A::MatrixArray)

#     # for j in eachindex(A)
#     #     A[j] = real.(A[j])
#     # end
#     # return A
#     #return MatrixArray(real(parent(A)))

#     # perhaps not performant but works
#     return AlgebraicArray(real.(Matrix(A)),rangedims(A), domaindims(A))
# end


# """
# function Matrix(P::MatrixArray{T}) where T
# """
# function Matrix(P::MatrixArray{T}) where T
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
# Base.transpose(P::MatrixArray) = AlgebraicArray( transpose(Matrix(P)), domaindims(P), rangedims(P))
# Base.adjoint(P::MatrixArray) = AlgebraicArray( adjoint(Matrix(P)), domaindims(P), rangedims(P))
# Base.similar(P::MatrixArray) = AlgebraicArray( similar(Matrix(P)), rangedims(P), domaindims(P))

# # CR format for multiplication
# # function Base.:*(A::MatrixArray, b::VectorArray)
# #     c = zero(first(A))
# #     for j in eachindex(A)
# #         c += A[j] * b[j]
# #     end
# #     return VectorArray(c)
# # end

# # slightly faster version in a one-liner form
# Base.:*(A::MatrixArray, b::VectorArray) =  AlgebraicArray(Matrix(A) * vec(b), rangedims(A))
# Base.:*(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) * Matrix(B), rangedims(A), domaindims(B))
# Base.:*(a::VectorArray, B::MatrixArray) = AlgebraicArray(vec(a) * Matrix(B), rangedims(a), domaindims(B))
# Base.:*(a::Number, b::VectorArray) = AlgebraicArray(a * vec(b), rangedims(b))
# Base.:*(b::VectorArray, a::Number) = a * b
# Base.:*(a::Number, B::MatrixArray) = AlgebraicArray(a * Matrix(B), rangedims(B), domaindims(B))
# Base.:*(B::MatrixArray, a::Number) = a * B

# Base.:(\ )(A::MatrixArray, b::VectorArray) = AlgebraicArray(Matrix(A) \ vec(b), domaindims(A))
# Base.:(\ )(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) \ Matrix(B), domaindims(A), domaindims(B))
# Base.:(/)(A::MatrixArray, b::Number) = AlgebraicArray(Matrix(A)/b, rangedims(A), domaindims(A))
# #     (c isa Number) && (c = [c]) # useful snippet if one-linear fails in some cases

# Base.:+(A::MatrixArray, B::MatrixArray) = MatrixArray(parent(A) + parent(B))
# Base.:+(a::VectorArray, b::VectorArray) = VectorArray(parent(a) + parent(b))
# Base.:+(A::MatrixArray, B::VectorArray) = MatrixArray(Matrix(A) + vec(B), rangedims(A), domaindims(A))
# Base.:+(A::VectorArray, B::MatrixArray) = B + A

# # special case: A is a wrapped scalar
# # certainly not performant, but this is just a 1x1 matrix
# function Base.:+(A::MatrixArray, b::Number)
#     # a wrapped scalar 
#     if (prod(domaindims(A)) ==1) && (prod(rangedims(A)) == 1)
#         return MatrixArray([first(first(A)) + b;;], rangedims(A), domaindims(A))
#     else
#         error("Matrix and scalar addition only possible with 1x1 Matrix")
#     end 
# end 

# Base.:-(A::MatrixArray, B::MatrixArray) = MatrixArray(parent(A) - parent(B))
# Base.:-(a::VectorArray, b::VectorArray) = VectorArray(parent(a) - parent(b))
# Base.:-(A::MatrixArray) = -1 * A
# # special case: A is a wrapped scalar
# # certainly not performant, but this is just a 1x1 matrix
# function Base.:-(A::MatrixArray, b::Number)
#     # a wrapped scalar 
#     if (prod(domaindims(A)) ==1) && (prod(rangedims(A)) == 1)
#         return MatrixArray([first(first(A)) - b;;], rangedims(A), domaindims(A))
#     else
#         error("Matrix and scalar subtraction only possible with 1x1 Matrix")
#     end 
# end 

# """
# function matrix right divide

# `A/B = ( B'\\A')'
# """
# Base.:(/)(A::MatrixArray, B::MatrixArray) = AlgebraicArray(Matrix(A) / Matrix(B), rangedims(A), rangedims(B))
# Base.:(/)(A::Union{VectorArray,MatrixArray}, b::Number) = (1/b) * A


# function LinearAlgebra.eigen(A::MatrixArray)
#     F = eigen(Matrix(A))
#     dsize = length(F.values)
#     rsize = rangedims(A)
#     values = AlgebraicArray(F.values,dsize)
#     vectors = AlgebraicArray(F.vectors,rsize,dsize) 
#     return Eigen(values, vectors)
# end

# # force it to return a MatrixArray
# Diagonal(a::VectorArray) = MatrixArray(Diagonal(vec(a)), rangedims(a), rangedims(a))

# function exp(A::MatrixArray)
#     # A must be endomorphic (check type signature someday)
#     !AlgebraicArrays.endomorphic(A) && error("A must be endomorphic to be consistent with matrix exponential")
#     eA = exp(Matrix(A)) # move upstream to MultipliableDimArrays eventually
#     return AlgebraicArray(exp(Matrix(A)),rangedims(A),domaindims(A)) # wrap with same labels and format as A
# end

# #########

# #struct VectorArray{T<:Number,N,A<:AbstractArray{T,N}} <: AbstractArray{T,1}
# struct VectorArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,1}
#     data:: A
# end
# VectorArray(a::Number) = a # helpful for slices that aren't vectors anymore

# # force a VectorArray if really needed
# function VectorArray(A::AbstractVector, rsize::Union{Int,NTuple{N,Int}}) where N
#     return VectorArray(reshape(A,rsize))
# end


end # module AlgebraicArrays
