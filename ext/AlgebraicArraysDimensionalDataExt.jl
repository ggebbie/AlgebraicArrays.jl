module AlgebraicArraysDimensionalDataExt

using AlgebraicArrays
using DimensionalData
using DimensionalData:@dim
using LinearAlgebra

export VectorDimArray, MatrixDimArray, AlgebraicDimArray
export dims, rowvector, AlgebraicArray
export rand, randn, zeros, ones

import AlgebraicArrays: rangedims, domaindims, AlgebraicArray
import AlgebraicArrays: MatrixArray, VectorArray 
import LinearAlgebra: eigen
import Base: exp, transpose
import Base: rand, randn, zeros, ones, fill
import DimensionalData: dims

@dim RowVector "singular dimension"
@dim Eigenmode "eigenmode"

MatrixDimArray = MatrixArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
VectorDimArray = VectorArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
AlgebraicDimArray = AlgebraicArray{T, D, N, A} where {T, D, N, A<:AbstractDimArray{T, N}}

# careful: these function conflict with ones in main module
rangedims(A::VectorDimArray) = dims(parent(A))
function rangedims(A::MatrixDimArray)
    Nrange = length(first(A.dims))
    return dims(parent(A))[1:Nrange]
end

function domaindims(A::MatrixDimArray)
    Nrange = length(first(A.dims))
    Ndomain = length(last(A.dims))
    return dims(parent(A))[Nrange+1:Nrange+Ndomain]
end

domaindims(b::VectorDimArray) = ()

DimensionalData.dims(A::VectorDimArray) = dims(parent(A))

# ### fill
function Base.fill(val, ddims::Union{Tuple, DD}, adims::NTuple{AD,Tuple}) where AD where DD <: DimensionalData.Dimension 
    return AlgebraicArray(fill(val, ddims), adims)
end

function Base.ones(ddims::Union{Tuple, DD}, adims::NTuple{AD,Tuple}) where AD where DD <: DimensionalData.Dimension 
    return AlgebraicArray(ones(ddims), adims)
end

function Base.zeros(ddims::Union{Tuple, DD}, adims::NTuple{AD,Tuple}) where AD where DD <: DimensionalData.Dimension 
    return AlgebraicArray(zeros(ddims), adims)
end

function Base.rand(ddims::Union{Tuple, DD}, adims::NTuple{AD,Tuple}) where AD where DD <: DimensionalData.Dimension 
    return AlgebraicArray(rand(ddims), adims)
end

function Base.randn(ddims::Union{Tuple, DD}, adims::NTuple{AD,Tuple}) where AD where DD <: DimensionalData.Dimension 
    return AlgebraicArray(DimArray(randn(size(ddims)), ddims), adims)
end

function Base.transpose(b::VectorDimArray)

    size_rowvector = ((1,),first(b.dims))
    arr = reshape(transpose(vec(b)),
                  AlgebraicArrays.unwrap(size_rowvector)...)
    newdim = (RowVector(["1"]), b.data.dims...)
    da = DimArray(arr, newdim) 
    return AlgebraicArray(da, size_rowvector)
end
function Base.transpose(P::MatrixDimArray)
    size_transpose = (last(P.dims),first(P.dims))
    arr = reshape(transpose(AlgebraicArrays.matrix_or_vec(P)),
                  AlgebraicArrays.unwrap(size_transpose)...)
    newdim = (domaindims(P)...,rangedims(P)...)
    da = DimArray(arr, newdim) 
    return AlgebraicArray(da, size_transpose)
end

Base.:*(a::Number,B::AlgebraicDimArray) = AlgebraicArray(a*parent(B),B.dims)
function Base.:*(A::AlgebraicDimArray, b::AlgebraicDimArray)
    if rangedims(b) == domaindims(A)
        if isempty(domaindims(b))
            newdim = rangedims(A)
            arr = reshape( AlgebraicArrays.matrix_or_vec(A)*AlgebraicArrays.matrix_or_vec(b), size(newdim)...)               
            da = DimArray(arr, newdim) 
            return AlgebraicArray(da, (size(newdim),))
        else
            newdim = (rangedims(A)...,domaindims(b)...)
            arr = reshape( AlgebraicArrays.matrix_or_vec(A)*AlgebraicArrays.matrix_or_vec(b), size(newdim)...)               
            da = DimArray(arr, newdim) 
            return AlgebraicArray(da, (size(rangedims(A)),size(domaindims(b))))
        end
    else
        error("multiplication with `AlgebraicArray`s not conformable")
    end
end

function Base.:(\ )(A::AlgebraicDimArray, B::AlgebraicDimArray) 
    (rangedims(A) !== rangedims(B)) && (error("AlgebraicArrays.jl: left divide not conformable"))
    if isempty(domaindims(B))
        newdim = domaindims(A)
        arr = reshape( AlgebraicArrays.matrix_or_vec(A) \
                       AlgebraicArrays.matrix_or_vec(B), size(newdim)...)
        da = DimArray(arr, newdim)
        return AlgebraicArray(da, (size(domaindims(A)), ))
    else
        newdim = (domaindims(A)...,domaindims(B)...)
        arr = reshape(
            AlgebraicArrays.matrix_or_vec(A)\AlgebraicArrays.matrix_or_vec(B),
            size(newdim)...)               
        da = DimArray(arr, newdim) 
        return AlgebraicArray(da, (size(domaindims(A)),size(domaindims(B))))
    end
end

# missing a compatibility test
function Base.:(/)(A::AlgebraicDimArray, B::AlgebraicDimArray)
    if isempty(rangedims(B))
        newdim = rangedims(A)
        arr = reshape(AlgebraicArrays.matrix_or_vec(A) /
                      AlgebraicArrays.matrix_or_vec(B), size(newdim)...)
        da = DimArray(arr, newdim)
        return AlgebraicArray(da, (size(rangedims(A)),))
    else
        newdim = (rangedims(A)..., rangedims(B)...)
        arr = reshape(AlgebraicArrays.matrix_or_vec(A) /
                      AlgebraicArrays.matrix_or_vec(B), size(newdim)...)
        da = DimArray(arr, newdim)
        return AlgebraicArray(da, (size(rangedims(A)),size(rangedims(B))))
    end
end

function LinearAlgebra.Diagonal(a::VectorDimArray) 
    newdim = AlgebraicArrays.unwrap((rangedims(a), rangedims(a)))
    arr = reshape( Diagonal(vec(a)), size(newdim))
    da = DimArray(arr, newdim)
    return AlgebraicArray(da, (size(rangedims(a)), size(rangedims(a))))
end

function Base.similar(aa::AlgebraicDimArray{T}) where T
    tmp = reshape(similar(Array{T}, axes(aa)), 
                  AlgebraicArrays.unwrap(aa.dims))
    da = DimArray(tmp, aa.data.dims)
    return AlgebraicArray(da, aa.dims)
end

Base.BroadcastStyle(::Type{<:AlgebraicDimArray}) = Broadcast.ArrayStyle{AlgebraicDimArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{AlgebraicDimArray}}, ::Type{ElType}) where ElType
    # Scan the inputs, first AArray amongst the arguments
    A = find_aa(bc)
    tmp = reshape(similar(Array{ElType}, axes(A)), 
        AlgebraicArrays.unwrap(A.dims))
    da = DimArray(tmp, A.data.dims)
    aa = AlgebraicArray( da, A.dims)
    return aa
end

# function Base.similar(aa::AlgebraicArray{T}) where T 
#     tmp = reshape(similar(Array{T}, axes(aa)), 
#         AlgebraicArrays.unwrap(aa.dims))
#     return AlgebraicArray(tmp, aa.dims)
# end

# "`A = find_va(As)` returns the first AlgebraicArray among the arguments."
find_aa(bc::Base.Broadcast.Broadcasted) = find_aa(bc.args)
find_aa(args::Tuple) = find_aa(find_aa(args[1]), Base.tail(args))
find_aa(x) = x
find_aa(::Tuple{}) = nothing
find_aa(a::AlgebraicArray, rest) = a
find_aa(::Any, rest) = find_aa(rest)

function LinearAlgebra.diag(A::MatrixDimArray)
    if endomorphic(A)

        size_range = size(rangedims(A))
        da = DimArray( reshape(diag(Matrix(A)), size_range), rangedims(A))
        return AlgebraicArray(da, (size_range,))
    else
        # unclear what to do about dimensions in this case
        # punt and return a vector, warning: type unstable
        return diag(Matrix(A))
    end
end 

# complete copy of DimData extension method to avoid dispatch ambiguity
# must be another way to disambiguate
function LinearAlgebra.eigen(A::MatrixDimArray)
    !endomorphic(A) && error("AlgebraicArrays.jl: not endomorphic")
    F = eigen(Matrix(A))

    eigen_dims = Eigenmode(1:length(F.values))
    newdim = (rangedims(A)..., eigen_dims)
    varr = reshape(F.vectors, size(newdim)...)
    vda = DimArray(varr, newdim)
    rdims_new = size(rangedims(A))
    ddims_new = size(eigen_dims)

    vectors = AlgebraicArray(vda,(rdims_new,ddims_new))

    # arr = AlgebraicArray(F.values, (size(eigen_dims),))
    arr = AlgebraicArray(F.values, (ddims_new,))
    da = DimArray(arr, eigen_dims)
    values = AlgebraicArray(da, (size(eigen_dims),))

    return Eigen(values, vectors)
end

function Base.exp(A::MatrixDimArray)
    # A must be endomorphic (check type signature someday)
    !AlgebraicArrays.endomorphic(A) && error("A must be endomorphic to be consistent with matrix exponential")
    arr = reshape( exp(Matrix(A)), AlgebraicArrays.unwrap(A.dims))
    da = DimArray(arr, dims(parent(A)))
    return AlgebraicArray(da, A.dims)
end

function Base.getindex(A::MatrixDimArray, inds::Vararg{Tuple,2}) 
    # reshape A to a Matrix
    inds_full = AlgebraicArrays.unwrap(inds)
    tmp = getindex(parent(A), inds_full...)

    tmp isa Number && return tmp

    # find the size of each input dim

    # range space
    rdims_in = rangedims(A)
    rinds_in = first(inds)
    Nrange = 0 # dimension of range
    for i in eachindex(rdims_in)
        if rdims_in[i][rinds_in[i]] isa DimensionalData.Dimension
            Nrange += 1
        end
    end

    # domain space
    ddims_in = domaindims(A)
    dinds_in = last(inds)
    Ndomain = 0 # dimension of domain
    for i in eachindex(ddims_in)
        if ddims_in[i][dinds_in[i]] isa DimensionalData.Dimension
            Ndomain += 1
        end
    end
    
    if iszero(Ndomain)
        asize = (size(tmp),)
        
    elseif iszero(Nrange)
        asize = ((1,),size(tmp))

        # get the extra label
        arr = reshape(tmp, AlgebraicArrays.unwrap(asize))
        newdim = (RowVector(["1"]), dims(tmp)...)
        tmp = DimArray(arr, newdim)
    else
        asize = (size(tmp)[1:Nrange], size(tmp)[Nrange+1:Nrange+Ndomain]) 
    end
    
    return AlgebraicArray( tmp, asize)
end

end #module
