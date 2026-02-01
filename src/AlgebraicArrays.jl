module AlgebraicArrays

using LinearAlgebra: NumberArray
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
        if D > 2
            error("tensors not handled")
        end
        if isempty(last(y))
            # matrix should be dropped to a vector
            # in accordance with base julia
            Dnew = D - 1
            ynew = (first(y),)
        else
            Dnew = D
            ynew = y
        end
        N == Dnew ? need_reshape = true : need_reshape = false  # passing algebraic data
        if need_reshape  # passing algebraic data
            x2 = reshape(x, unwrap(ynew))
            return new{T,Dnew,ndims(x2)}(x2,ynew)
        else
            return new{T,Dnew,N}(x,ynew)
        end
    end
end

AlgebraicArray(a::Number, b) = a # helpful for slices that aren't vectors anymore


# size of data array
asize(A::AlgebraicArray) = unwrap(A.dims)

unwrap(d::NTuple{D,Tuple}) where D  =
    Tuple(d[i][j] for i in eachindex(d) for j in eachindex(d[i]))
# unwrap(d::Tuple)  =
#     Tuple(d[i][j] for i in eachindex(d) for j in eachindex(d[i]))

parent(A::AlgebraicArray) = A.data
Base.size(A::AlgebraicArray) = prod.(A.dims)

rangedims(A::AlgebraicArray) = first(A.dims)
endomorphic(A::AlgebraicArray) = isequal(rangedims(A), domaindims(A))

# subset of all AArrays is a VArray (VectorArray)
VectorArray{T,N} = AlgebraicArray{T,1,N}

# don't return as AArray
# force only one argument for this vector
# function Base.getindex(b::VectorArray, ind::Vararg{Any,1})
#     return tmp =  getindex(b.data, ind)
function Base.getindex(b::VectorArray, inds::Vararg)
    tmp =  getindex(b.data, inds...)
    # newsize = (size(tmp),)
    return AlgebraicArray( tmp, (size(tmp),))
end
# do return as AArray
function Base.getindex(A::VectorArray, inds::Tuple)
    # inds_full = unwrap(inds)
    tmp = getindex(A.data, inds...)
    return AlgebraicArray( tmp, (size(tmp),))
end

# Base.getindex(b::VectorArray, inds::Vararg) = VectorArray(getindex(parent(b), inds...))
# Base.getindex(b::VectorArray; kw...) = VectorArray(getindex(parent(b); kw...))
# Base.getindex(b::VectorArray; kw...) = getindex(parent(b); kw...)
Base.getindex(b::VectorArray; kw...) = getindex(vec(b); kw...)
# Base.getindex(b::VectorArray; ind::Tuple) = getindex(parent(b), ind...)

Base.setindex!(b::VectorArray, val, inds::Vararg) = b.data[inds...] = val
# function Base.setindex!(b::VectorArray, val, inds::Tuple)
#      b.data[inds...] = val
# end

Base.vec(b::VectorArray) = vec(b.data)
matrix_or_vec(b::VectorArray) = vec(b.data)

function Base.show(io::IO, mime::MIME"text/plain", b::AlgebraicArray)
    #println(summary(b))
    show(io,mime,parent(b))
    println(io,"")
    println(io,"============================")
    println(io,"*operating algebraically as*")
    show(io,mime,matrix_or_vec(b))
end

domaindims(q::VectorArray) = ()

# Base.transpose(q::VectorArray) =
#     AlgebraicArray( transpose(vec(q)), (domaindims(q), rangedims(q)))

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

# # # implement broadcast
# when I add this, broadcast over entire MatrixArray doesn't work
# Base.IndexStyle(A::AlgebraicArray) = Base.IndexStyle(parent(A))

Base.BroadcastStyle(::Type{<:AlgebraicArray}) = Broadcast.ArrayStyle{AlgebraicArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{AlgebraicArray}}, ::Type{ElType}) where ElType
    # Scan the inputs
    A = find_aa(bc)
    AlgebraicArray(similar(Array{ElType}, axes(bc)), A.dims)
end
function Base.similar(aa::AlgebraicArray{T}) where T 
    tmp = reshape(similar(Array{T}, axes(aa)), 
        AlgebraicArrays.unwrap(aa.dims))
    return AlgebraicArray(tmp, aa.dims)
end

# "`A = find_va(As)` returns the first AlgebraicArray among the arguments."
find_aa(bc::Base.Broadcast.Broadcasted) = find_aa(bc.args)
find_aa(args::Tuple) = find_aa(find_aa(args[1]), Base.tail(args))
find_aa(x) = x
find_aa(::Tuple{}) = nothing
find_aa(a::AlgebraicArray, rest) = a
find_aa(::Any, rest) = find_aa(rest)

# subset of all AArrays is a MArray (MatrixArray)
MatrixArray{T,N} = AlgebraicArray{T,2,N}

# don't return AArray
function Base.getindex(A::MatrixArray, inds::Vararg{Any,2})
    # reshape A to a Matrix
    A2 = reshape(A.data, size(A))
    return tmp = getindex(A2, inds...)
    # Nrow = length(first(D.dims))
    # Ncol = length(last(D.dims))
    # fsize = size(tmp)
    # asize = Tuple( Tuple(fsize[1:Nrow]), Tuple(fsize[Nrow+1:Nrow+Ncol]))
    # return AlgebraicArray( tmp, asize)
end
# do return AArray
function Base.getindex(A::MatrixArray, inds::Vararg{Tuple,2})
    # reshape A to a Matrix
    inds_full = unwrap(inds)
    tmp = getindex(parent(A), inds_full...)

    tmp isa Number && return tmp

    Nrow = length(first(A.dims))
    Ncol = length(last(A.dims))

    # how many singleton dimensions have dropped out?
    Nrowdrop = count(isa.(inds_full[1:Nrow],Integer))
    Ncoldrop = count(isa.(inds_full[Nrow+1:Nrow+Ncol],Integer))

    fsize = size(tmp)
    Nrow_new = Nrow - Nrowdrop
    Ncol_new = Ncol - Ncoldrop

    if iszero(Ncol_new)
        asize = ((fsize[1:Nrow_new]),)
    else
        asize = (fsize[1:Nrow_new],fsize[Nrow_new+1:Nrow_new+Ncol_new])
    end
    
    return AlgebraicArray( tmp, asize)
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
# function Base.show(io::IO, mime::MIME"text/plain", A::MatrixArray)
#     show(io,mime,parent(A))
#     println(io,"")
#     println(io,"============================")
#     println(io,"*operating algebraically as*")
#     show(io,mime,Matrix(A))
# end

# Base.size(A::MatrixArray) = size(parent(A))
Base.Matrix(P::MatrixArray) = reshape( P.data, size(P))
matrix_or_vec(P::MatrixArray) = reshape( P.data, size(P))

# function Base.getindex(A::MatrixArray, inds::Vararg)
#     Aslice = getindex(parent(A), inds...)
#     return AlgebraicArray(Aslice)
# end

# rowvector(A::MatrixArray, rowindex::Vararg) = transpose(VectorArray([A[j][rowindex...] for j in eachindex(A)]))
    
# Base.getindex(A::MatrixArray; kw...) = getindex(parent(A), kw...) 

# set this up for algebraic and dimensional layouts
function Base.setindex!(MA::MatrixArray, val, inds::Vararg)
    # Matrix step wicked slow?
    setindex!(Matrix(MA), val, inds...)
end

# Base.setindex!(A::MatrixArray, v, inds::Vararg) = setindex!(parent(A), v, inds...) # need to reverse order?
# Base.setindex!(A::MatrixArray, v; kw...) = setindex!(parent(A), v, kw...) 
# #Base.IndexStyle(A::MatrixArray) = Base.IndexStyle(parent(A))

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

domaindims(P::MatrixArray) = last(P.dims)

# # a pattern for any function
Base.transpose(P::AlgebraicArray) =
    AlgebraicArray( transpose(matrix_or_vec(P)), (domaindims(P), rangedims(P)))

Base.adjoint(P::AlgebraicArray) =
    AlgebraicArray( adjoint(matrix_or_vec(P)), (domaindims(P), rangedims(P)))

Base.:(\ )(A::AlgebraicArray, B::AlgebraicArray) =
    (rangedims(A) == rangedims(B)) ? 
    (return AlgebraicArray(matrix_or_vec(A) \ matrix_or_vec(B), (domaindims(A), domaindims(B)))) :
    (error("AlgebraicArrays.jl: left divide not conformable"))

# needed?
# Base.:(/)(A::MatrixArray, b::Number) = AlgebraicArray(Matrix(A)/b, rangedims(A), domaindims(A))

# missing an explicit conformability test here
Base.:(/)(A::AlgebraicArray, B::AlgebraicArray) = AlgebraicArray( matrix_or_vec(A) / matrix_or_vec(B), (rangedims(A), rangedims(B)))

# Base.:(/)(A::Union{VectorArray,MatrixArray}, b::Number) = (1/b) * A


# Base.similar(P::MatrixArray) = AlgebraicArray( similar(Matrix(P)), rangedims(P), domaindims(P))

function Base.:*(A::AlgebraicArray, b::AlgebraicArray)
    if rangedims(b) == domaindims(A)
        return AlgebraicArray(matrix_or_vec(A)*matrix_or_vec(b), (rangedims(A), domaindims(b)))
    else
        error("multiplication with `AlgebraicArray`s not conformable")
    end
end

function LinearAlgebra.eigen(A::MatrixArray)
    !endomorphic(A) && error("AlgebraicArrays.jl: not endomorphic")
    F = eigen(Matrix(A))
    dsize = size(F.values)
    rsize = rangedims(A)
    values = AlgebraicArray(F.values,(dsize,))
    vectors = AlgebraicArray(F.vectors,(rsize,dsize)) 
    return Eigen(values, vectors)
end

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
