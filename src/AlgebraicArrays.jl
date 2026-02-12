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
struct AlgebraicArray{T,D,N,A <: AbstractArray{T,N}} <: AbstractArray{T,D}
    data:: A
    dims:: NTuple{D,Tuple}
    function AlgebraicArray(x::A,y::NTuple{D,Tuple}) where A <: AbstractArray{T,N} where {T,D,N} 
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
            println("needs reshape")
            x2 = reshape(x, unwrap(ynew))
            return new{T,Dnew,ndims(x2),typeof(x2)}(x2,ynew)
        else
            return new{T,Dnew,N,A}(x,ynew)
        end
    end
end

AlgebraicArray(a::Number, b) = a # helpful for slices that aren't vectors anymore

# size of data array
asize(A::AlgebraicArray) = unwrap(A.dims)

unwrap(d::NTuple{D,Tuple}) where D  =
    Tuple(d[i][j] for i in eachindex(d) for j in eachindex(d[i]))

parent(A::AlgebraicArray) = A.data
Base.size(A::AlgebraicArray) = prod.(A.dims)

rangedims(A::AlgebraicArray) = first(A.dims)
endomorphic(A::AlgebraicArray) = isequal(rangedims(A), domaindims(A))

# subset of all AArrays is a VArray (VectorArray)
VectorArray{T,N,A} = AlgebraicArray{T,1,N,A}

function Base.getindex(b::VectorArray, inds::Vararg)
    tmp =  getindex(b.data, inds...)
    return AlgebraicArray( tmp, (size(tmp),))
end
function Base.getindex(A::VectorArray, inds::Tuple)
    tmp = getindex(A.data, inds...)
    return AlgebraicArray( tmp, (size(tmp),))
end

Base.getindex(b::VectorArray; kw...) = getindex(vec(b); kw...)
Base.setindex!(b::VectorArray, val, inds::Vararg) = b.data[inds...] = val

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
MatrixArray{T,N,A} = AlgebraicArray{T,2,N,A}

# don't return AArray
function Base.getindex(A::MatrixArray, inds::Vararg{Any,2})
    # reshape A to a Matrix
    A2 = reshape(A.data, size(A))
    return tmp = getindex(A2, inds...)
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

Base.Matrix(P::MatrixArray) = reshape( P.data, size(P))
matrix_or_vec(P::MatrixArray) = reshape( P.data, size(P))

# set this up for algebraic and dimensional layouts
function Base.setindex!(MA::MatrixArray, val, inds::Vararg{Any,2})
    # Matrix step wicked slow?
    setindex!(Matrix(MA), val, inds...)
end
function Base.setindex!(MA::MatrixArray, val, inds::Vararg{Tuple,2})
    inds_full = unwrap(inds)
    setindex!(parent(MA), val, inds_full...)
end

function LinearAlgebra.diag(A::MatrixArray)
    if endomorphic(A)
        return AlgebraicArray(diag(Matrix(A)),(rangedims(A),))
    else
        # unclear what to do about dimensions in this case
        # punt and return a vector, warning: type unstable
        return diag(Matrix(A))
    end
end 

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

# # force `Diagonal` to return an AlgebraicArray
LinearAlgebra.Diagonal(a::VectorArray) = AlgebraicArray(Diagonal(vec(a)), (rangedims(a), rangedims(a)))

function exp(A::MatrixArray)
    # A must be endomorphic (check type signature someday)
    !AlgebraicArrays.endomorphic(A) && error("A must be endomorphic to be consistent with matrix exponential")
    eA = exp(Matrix(A))
    # wrap with same labels and format as A
    return AlgebraicArray(exp(Matrix(A)),(rangedims(A),domaindims(A)))
end

end # module AlgebraicArrays
