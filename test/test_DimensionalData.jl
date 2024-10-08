using Revise
using AlgebraicArrays
#using Unitful
#using UnitfulLinearAlgebra
using DimensionalData
using DimensionalData:@dim
using Test

# fixed parameters
@dim YearCE "years Common Era"
@dim SurfaceRegion "surface location"
@dim InteriorLocation "interior location"
@dim StateVariable "state variable"
surfaceregions = [:NATL,:ANT,:SUBANT]
N = length(surfaceregions)
years = (1990:1993)
statevariables = [:θ, :δ¹⁸O] 
M = 5 # Interior Locations with obs

function source_water_solution(surfaceregions,years)
    m = length(years)
    n = length(surfaceregions)
    return VectorArray(DimArray(randn(m,n),(Ti(years),SurfaceRegion(surfaceregions))))
end

function source_water_solution(surfaceregions, years, statevar)
    m = length(years)
    n = length(surfaceregions)
    mat = cat(randn(m, n, 1), randn(m, n, 1); dims = 3)
    x = VectorArray(DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar))))
    return x
end

@testset "AlgebraicArrays + DimensionalData.jl" begin

    @testset "no units" begin
        x = source_water_solution(surfaceregions,
            years,
            statevariables);

        x = source_water_solution(surfaceregions, years);

        # test that these vectors;matrices can be used in algebraic expressions
        y = vec(x)
        z = AlgebraicArray(y, dims(parent(x)))
        @test x == z

        # make the diagonal elements
        v = ones(dims(parent(x)))
        w = VectorArray(v) 

        D = Diagonal(w)
        DT = transpose(D)
        DTT = transpose(DT)
        @test D == DT
        @test D == DTT

        R = AlgebraicArray(rand(length(x),length(x)),
            rangesize(x), rangesize(x))    
        RT = transpose(R)
        RTT = transpose(RT)
        @test R == RTT
        @test R ≠ RT
        
        q = R * x
        @test q isa VectorArray{T,N,DA} where T where N where DA <: DimensionalData.AbstractDimArray
        
        y = R \ q
        @test isapprox(x, y, atol = 1e-8)

        S = AlgebraicArray(rand(length(x),length(x)),
            rangesize(x), rangesize(x))    
        Q = R * S
        U = R \ Q 
        @test isapprox(Matrix(U), Matrix(S), atol = 1e-8)

        # square matrices, matrix matrix right divide
        @test isapprox(Matrix(Q / S), Matrix(R), atol = 1e-8)
        
    end

    @testset "eigenstructure" begin 
        x = source_water_solution(surfaceregions,
            years,
            statevariables)
        S = AlgebraicArray(rand(length(x),length(x)),
            rangesize(x), rangesize(x))    

        D, V = eigen(S)
        F = eigen(S)

        Sx_eigen = V * D / V
        @test isapprox(Sx, Sx_eigen, atol = 1e-8)

        # check matrix exponential
        @test MultipliableDimArrays.endomorphic(Sx)
        exp(Sx) # watch out for overflow!
    end

    @testset "UnitfulLinearAlgebra extension" begin

        using Unitful
        
        function source_water_solution_with_units(surfaceregions, years, statevar)
            yr = u"yr"
            K = u"K"
            permil = u"permille"
            m = length(years)
            n = length(surfaceregions)
            mat = cat(randn(m, n, 1)K, randn(m, n, 1)permil; dims = 3)
            x = DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar)))
            return x
        end

        x = source_water_solution_with_units(surfaceregions,
            years,
            statevariables)

        # test that these vectors;matrices can be used in algebraic expressions
        xvec = vec(x)
        x3 = MultipliableDimArray(xvec, dims(x))
        @test x == x3

#        using UnitfulLinearAlgebra

        D = MultipliableDimArray(randn(length(x),length(x)),dims(x),dims(x))
        urange = unit.(x)
        udomain = unit.(inv.(x))
        Px = UnitfulMatrix(D,(urange,udomain))

        # try to remove reference 
        Pxmat = MultipliableDimArrays.Matrix(Px)
        Px2 = MultipliableDimArray(Pxmat, dims(x), dims(x))
        @test Px[2][3] == ustrip(Px2[2][3]) # weak test, but function to revert operation is not completed, one is DimArray, one is UnitfulMatrix

        PxT = transpose(Px)
        PxTT = transpose(PxT)
#        @test Px == PxTT
        @test Px[2][3] == ustrip(PxTT[2][3]) # weak test, but function to revert operation is not completed, one is DimArr    
        # make quick inverse of Px for unit compatibility
        iPx = UnitfulMatrix(D,(udomain,urange))

        # matrix-vector multiplication
        q = iPx * x

        x2 = iPx \ q
        @test isapprox(ustrip.(x), ustrip.(x2), atol = 1e-5)

        Ialmost = iPx * Px # matrix-matrix multiplication, except iPx is not actually the inverse of Px
        MultipliableDimArrays.Matrix(Ialmost)  # visually reasonable

        # does matrix-matrix left divide work?
        Px2 = iPx \ Ialmost
        @test isapprox(Px[2][3], ustrip(Px2[2][3]), atol = 1e-8) # weak test, but function to revert operation is not completed, one is DimArr    
        @test isapprox(Px, Px2, atol = 1e-8) # weak test, but function to revert operation is not comend
    end
end
