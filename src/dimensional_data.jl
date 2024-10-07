@dim RowVector "singular dimension"

# put dimensional info into size, change name?
function rangesize(b::VectorArray{T,N,DA}) where T where N where DA <: DimensionalData.AbstractDimArray

    return dims(parent(b))
end

domainsize(b::VectorArray{T,N,DA}) where T where N where DA <: DimensionalData.AbstractDimArray = ()

# would prefer to be more specific about the type of Tuple
# instead I made the core routines dispatch with a specific Tuple structure
function AlgebraicArray(A::AbstractVector, rdims::Union{Tuple,D}) where D <: DimensionalData.Dimension
#function AlgebraicArray(A::AbstractVector, rdims::Tuple)
    rsize = size(rdims)
    M = prod(rsize)
    if M > 1
        return VectorArray(DimArray(reshape(A,rsize),rdims))
    elseif M == 1
        # warning: introduces type instability
        # but useful for inner products
        return first(A)
    end
end

function AlgebraicArray(A::AbstractMatrix{T}, rdims::Union{Tuple,D1}, ddims::Union{Tuple,D2}) where T where D1 <: DimensionalData.Dimension where D2 <: DimensionalData.Dimension
#function AlgebraicArray(A::AbstractVector, rdims::Tuple)
    rsize = size(rdims)
    dsize = size(ddims)
    M = prod(dsize)
    N = length(rsize)

    if M > 1
        P = Array{Array{T,N}}(undef,dsize)
        for j in 1:M 
            P[j] = DimArray(reshape(A[:,j],rsize),rdims)
        end
        return MatrixArray(P)
    elseif M == 1
        # warning: introduces type instability
        # but useful for transpose of row vector
        return VectorArray(DimArray(reshape(A,rsize),rdims))
    else
        error("incompatible number of columns") 
    end
end

#    ones_row_vector = MultipliableDimArray(ones(1,2),,dims(B))
function Base.transpose(b::VectorArray{T,N,DA}) where T<:Number where N where DA <: DimensionalData.AbstractDimArray

    return AlgebraicArray(transpose(vec(b)), RowVector(["1"]), rangesize(b))
end
