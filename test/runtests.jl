using Revise
using AlgebraicArrays
using LinearAlgebra
using Test
using DimensionalData
using DimensionalData:@dim
using Unitful

# fixed parameters
@dim YearCE "years Common Era"
@dim SurfaceRegion "surface location"
@dim InteriorLocation "interior location"
@dim StateVariable "state variable"

@testset "AlgebraicArrays.jl" begin
    include("test_AlgebraicArrays.jl")
    include("test_DimensionalData.jl")
    include("test_unitful.jl")
    include("test_DimensionalData_Unitful.jl")
end
