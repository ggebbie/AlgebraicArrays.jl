module AlgebraicArraysDimensionalDataExt

using AlgebraicArrays
using DimensionalData
using DimensionalData:@dim
using LinearAlgebra

export VectorDimArray, MatrixDimArray, dims

import AlgebraicArrays: rangesize, domainsize, AlgebraicArray
import LinearAlgebra: eigen
import Base: exp, transpose
import DimensionalData: dims

@dim RowVector "singular dimension"
@dim Eigenmode "eigenmode"

MatrixDimArray = MatrixArray{T, M, N, R} where {M, T, N, R<:AbstractDimArray{T, M}}
VectorDimArray = VectorArray{T,N,A} where {T, N, A <: DimensionalData.AbstractDimArray}

rangesize(A::Union{VectorDimArray,MatrixDimArray}) = dims(parent(A))

domainsize(b::VectorDimArray) = ()
domainsize(A::MatrixDimArray) = dims(first(parent(A)))
DimensionalData.dims(A::VectorDimArray) = dims(parent(A))

# implement broadcast
Base.BroadcastStyle(::Type{<:VectorDimArray}) = Broadcast.ArrayStyle{VectorDimArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorDimArray}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ArrayAndChar:
    A = find_vda(bc)
    # Use the char field of A to create the output
    #WrappedDimArray(similar(Array{ElType}, axes(bc))) #, A.char)
    VectorArray(DimArray(similar(Array{ElType}, axes(bc)), dims(A)))
end
function Base.similar(vda::VectorDimArray{T}) where T
    VectorArray(similar(parent(vda))) #Array{T}, axes(vda)), dims(vda))
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
        return VectorArray(first(A)) # bugfix?
    end
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
        # Ptmp = DimArray(P,ddims)
        # println(typeof(Ptmp))
        # return MatrixArray(Ptmp)
        return MatrixArray(DimArray(P,ddims))
    elseif M == 1
        # warning: introduces type instability
        # but useful for transpose of row vector
        return VectorArray(DimArray(reshape(A,rsize),rdims))
    else
        error("incompatible number of columns") 
    end
end

Base.transpose(b::VectorDimArray) = AlgebraicArray(transpose(vec(b)), RowVector(["1"]), rangesize(b))

function  LinearAlgebra.eigen(A::MatrixDimArray)
    F = eigen(Matrix(A))
    #dsize = length(F.values)
    eigen_dims = Eigenmode(1:length(F.values))

    rsize = rangesize(A)
    values = AlgebraicArray(F.values, eigen_dims)
    vectors = AlgebraicArray(F.vectors,rsize,eigen_dims) 
    return Eigen(values, vectors)
end

function Base.exp(A::MatrixDimArray)
    # A must be endomorphic (check type signature someday)
    !endomorphic(A) && error("A must be endomorphic to be consistent with matrix exponential")
    eA = exp(Matrix(A)) # move upstream to MultipliableDimArrays eventually
    return AlgebraicArray(exp(Matrix(A)),rangesize(A),domainsize(A)) # wrap with same labels and format as A
end

end #module
