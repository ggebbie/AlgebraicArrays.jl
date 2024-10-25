@testset "Dimensional Data + Unitful" begin

    # fixed parameters
    @dim YearCE "years Common Era"
    @dim SurfaceRegion "surface location"
    @dim InteriorLocation "interior location"
    @dim StateVariable "state variable"
    surfaceregions = [:NATL,:ANT]
    N = length(surfaceregions)
    years = (1990:1992)
    statevariables = [:θ, :δ¹⁸O] 
    M = 2 # Interior Locations with obs

    function source_water_solution_with_units(surfaceregions, years, statevar)
        yr = u"yr"
        K = u"K"
        permil = u"permille"
        m = length(years)
        n = length(surfaceregions)
        mat = cat(randn(m, n, 1)K, randn(m, n, 1)permil; dims = 3)
        x = VectorArray(DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar))))
        return x
    end

    x = source_water_solution_with_units(surfaceregions,
        years,
        statevariables)

    # test that these vectors;matrices can be used in algebraic expressions
    y = vec(x)
    z = AlgebraicArray(y, dims(parent(x)))
    @test x == z

    # make the diagonal elements
    v = ones(dims(parent(x)))*u"J"
    w = VectorArray(v) 
    
    D = Diagonal(w)
    DT = transpose(D)
    DTT = transpose(DT)
    @test D == DT
    @test D == DTT

    R = AlgebraicArray(rand(length(x),length(x)),
        rangedims(x), rangedims(x))*u"mol"    
    RT = transpose(R)
    RTT = transpose(RT)
    @test R == RTT
    @test R ≠ RT

    q = R * x
    @test q isa VectorArray{T,N,DA} where T where N where DA <: DimensionalData.AbstractDimArray

    ##################################
    
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
