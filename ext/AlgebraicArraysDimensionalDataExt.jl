module AlgebraicArraysDimensionalDataExt

using AlgebraicArrays
using DimensionalData
using DimensionalData:@dim
using LinearAlgebra

export VectorDimArray, MatrixDimArray, dims, rowvector, AlgebraicArray
export rand, randn, zeros, ones

import AlgebraicArrays: rangedims, domaindims, AlgebraicArray, rowvector
import LinearAlgebra: eigen
import Base: exp, transpose
import Base: rand, randn, zeros, ones
import DimensionalData: dims

@dim RowVector "singular dimension"
@dim Eigenmode "eigenmode"

MatrixDimArray = MatrixArray{T, M, N, R} where {M, T, N, R<:AbstractDimArray{T, M}}
VectorDimArray = VectorArray{T, N, A} where {T, N, A <: DimensionalData.AbstractDimArray}

#VectorDimArray(array,rdims) = VectorArray(DimArray(array,rdims))
    
rangedims(A::VectorDimArray) = dims(parent(A))
rangedims(A::MatrixDimArray) = dims(first(parent(A)))

domaindims(A::MatrixDimArray) = dims(parent(A))
domaindims(b::VectorDimArray) = ()

DimensionalData.dims(A::VectorDimArray) = dims(parent(A))

# implement broadcast

# function declaration passes here but not OGFM
#Base.BroadcastStyle(::Type{<:VectorArray{T, N, A}}) where {T, N, A <: DimensionalData.AbstractDimArray} = Broadcast.ArrayStyle{VectorArray{T, N, A}}()
Base.BroadcastStyle(::Type{<:VectorArray{T, N, A}}) where {T, N, A <: DimensionalData.AbstractDimArray} = Broadcast.ArrayStyle{VectorDimArray}()

# passes test 
# Base.BroadcastStyle(::Type{<:VectorDimArray}) = Broadcast.ArrayStyle{VectorDimArray}()

#function opening passes here but not OGFM
#function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorArray{T, N, A}}}, ::Type{ElType}) where {ElType, T, N, A <: DimensionalData.AbstractDimArray}

#passes test
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorDimArray}}, ::Type{ElType}) where ElType
    B = find_vda(bc)
    VectorArray(DimArray(similar(Array{ElType}, axes(bc)), dims(B)))
end

function Base.similar(vda::VectorDimArray{T}) where T
    VectorArray(similar(parent(vda))) #Array{T}, axes(vda)), dims(vda))
end

### rand
function Base.rand(rdims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension
    if type == :VectorArray
        # actually use rand (randn not implemented for DimArray)
        return VectorArray(rand(rdims))
    else
        error("randn not implemented for this type")
    end
end
function Base.rand(T::Type, rdims::Union{Tuple,D}, ddims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension

    !(type == :MatrixArray || type == :AlgebraicArray ) && error("type not implemented")
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(rsize)
    N = prod(dsize)
    A = rand(M,N)
    return AlgebraicArray(A, rdims, ddims)
end
# make Float64 the default
Base.rand(rdims::Union{Tuple, D}, ddims::Union{Tuple, D}, type::Symbol) where D <: DimensionalData.Dimension =
    rand(Float64, rdims, ddims, type)

### randn
function Base.randn(rdims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension
    if type == :VectorArray
        return AlgebraicArray(randn(prod(size(rdims))), rdims)
    else
        error("randn not implemented for this type")
    end
end
function Base.randn(T::Type, rdims::Union{Tuple,D}, ddims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension

    !(type == :MatrixArray || type == :AlgebraicArray ) && error("type not implemented")
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(rsize)
    N = prod(dsize)
    A = randn(M,N)
    return AlgebraicArray(A, rdims, ddims)
end
# make Float64 the default
Base.randn(rdims::Union{Tuple, D}, ddims::Union{Tuple, D}, type::Symbol) where D <: DimensionalData.Dimension =
    randn(Float64, rdims, ddims, type)

### ones
function Base.ones(rdims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension
    if type == :VectorArray
        return AlgebraicArray(ones(prod(size(rdims))), rdims)
    else
        error("ones not implemented for this type")
    end
end
function Base.ones(T::Type, rdims::Union{Tuple,D}, ddims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension

    !(type == :MatrixArray || type == :AlgebraicArray ) && error("type not implemented")
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(rsize)
    N = prod(dsize)
    A = ones(M,N)
    return AlgebraicArray(A, rdims, ddims)
end
# make Float64 the default
Base.ones(rdims::Union{Tuple, D}, ddims::Union{Tuple, D}, type::Symbol) where D <: DimensionalData.Dimension =
    ones(Float64, rdims, ddims, type)

### zeros
function Base.zeros(rdims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension
    if type == :VectorArray
        return AlgebraicArray(zeros(prod(size(rdims))), rdims)
    else
        error("randn not implemented for this type")
    end
end
function Base.zeros(T::Type, rdims::Union{Tuple,D}, ddims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension

    !(type == :MatrixArray || type == :AlgebraicArray ) && error("type not implemented")
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(rsize)
    N = prod(dsize)
    A = zeros(M,N)
    return AlgebraicArray(A, rdims, ddims)
end
# make Float64 the default
Base.zeros(rdims::Union{Tuple, D}, ddims::Union{Tuple, D}, type::Symbol) where D <: DimensionalData.Dimension =
    zeros(Float64, rdims, ddims, type)

### fill
function Base.fill(val, rdims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension
    if type == :VectorArray
        return VectorArray(fill(val, rdims)) #AlgebraicArray(fill(prod(size(rdims))), rdims)
    else
        error("randn not implemented for this type")
    end
end
function Base.fill(val, rdims::Union{Tuple,D}, ddims::Union{Tuple,D}, type::Symbol) where D <: DimensionalData.Dimension

    !(type == :MatrixArray || type == :AlgebraicArray ) && error("type not implemented")
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(rsize)
    N = prod(dsize)
    A = fill(val, M, N)
    return AlgebraicArray(A, rdims, ddims)
end

"`A = find_vda(As)` returns the first VectorDimArray among the arguments."
find_vda(bc::Base.Broadcast.Broadcasted) = find_vda(bc.args)
find_vda(args::Tuple) = find_vda(find_vda(args[1]), Base.tail(args))
find_vda(x) = x
find_vda(::Tuple{}) = nothing
find_vda(a::VectorDimArray, rest) = a
find_vda(::Any, rest) = find_vda(rest)

# would prefer to be more specific about the type of Tuple
# instead I made the core routines dispatch with a specific Tuple structure
function AlgebraicArray(A::AbstractVector, rdims::Union{Tuple,D}) where D <: DimensionalData.Dimension
    rsize = size(rdims)
    M = prod(rsize)
    if M > 1
        return VectorArray(DimArray(reshape(A,rsize),rdims))
    elseif M == 1
        # warning: introduces type instability
        # but useful for inner products
        #return VectorArray(first(A)) # bugfix?
        return first(A) # bugfix?
    end
end

# force-construct a VectorArray if you know what you want
function VectorArray(A::AbstractVector, rdims::Union{Tuple,D}) where D <: DimensionalData.Dimension
    rsize = size(rdims)
    M = prod(rsize)
    return VectorArray(DimArray(reshape(A,rsize),rdims))
end

function AlgebraicArray(A::AbstractMatrix{T}, rdims::Union{Tuple,D1}, ddims::Union{Tuple,D2}) where T where D1 <: DimensionalData.Dimension where D2 <: DimensionalData.Dimension
#function AlgebraicArray(A::AbstractVector, rdims::Tuple)
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(dsize)
    N = length(rsize)

    if M > 1
        P = Array{DimArray{T,N}}(undef,dsize)
        for j in 1:M
            P[j] = DimArray(reshape(A[:,j],rsize),rdims)
        end
        return MatrixArray(DimArray(P,ddims))
    elseif M == 1
        # warning: introduces type instability
        # but useful for transpose of row vector
        return VectorArray(DimArray(reshape(A,rsize),rdims))
    else
        error("incompatible number of columns") 
    end
end

# force output to be `MatrixArray` even in funny/limiting cases
function MatrixArray(A::AbstractMatrix{T}, rdims::Union{Tuple,D1}, ddims::Union{Tuple,D2}) where T where D1 <: DimensionalData.Dimension where D2 <: DimensionalData.Dimension

    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(dsize)
    N = length(rsize)

    P = Array{DimArray{T,N}}(undef,dsize)
    for j in 1:M
        P[j] = DimArray(reshape(A[:,j],rsize),rdims)
    end
    return MatrixArray(DimArray(P,ddims))
end

Base.transpose(b::VectorDimArray) = AlgebraicArray(transpose(vec(b)), RowVector(["1"]), rangedims(b))

# undefined resource
# function Base.similar(mda::MatrixDimArray{T}) where T
#     MatrixArray(similar(parent(mda))) #Array{T}, axes(vda)), dims(vda))
# end

function  LinearAlgebra.eigen(A::MatrixDimArray)
    F = eigen(Matrix(A))
    #dsize = length(F.values)
    eigen_dims = Eigenmode(1:length(F.values))

    rsize = rangedims(A)
    values = AlgebraicArray(F.values, eigen_dims)
    vectors = AlgebraicArray(F.vectors,rsize,eigen_dims) 
    return Eigen(values, vectors)
end

function Base.exp(A::MatrixDimArray)
    # A must be endomorphic (check type signature someday)
    !AlgebraicArrays.endomorphic(A) && error("A must be endomorphic to be consistent with matrix exponential")
    eA = exp(Matrix(A)) # move upstream to MultipliableDimArrays eventually
    return AlgebraicArray(exp(Matrix(A)),rangedims(A),domaindims(A)) # wrap with same labels and format as A
end

#rowvector(A::MatrixDimArray{T,M,N}, rowindex::Vararg) where {T,M,N} = transpose(A[fill(:,N)...][rowindex...])
#rowvector(A::MatrixDimArray, rowindex::Vararg) = transpose(A[fill(:,N)...][rowindex...])
function rowvector(A::MatrixDimArray{T,M,N}, rowindex::Vararg) where {T,M,N}
    DA = DimArray([A[j][rowindex...] for j in eachindex(A)],domaindims(A))
    return transpose(VectorArray(DA))
end

end #module
