@testset "Dimensional Data" begin

    # fixed parameters
    @dim YearCE "years Common Era"
    @dim SurfaceRegion "surface location"
    @dim InteriorLocation "interior location"
    @dim StateVariable "state variable"
    #surfaceregions = [:NATL,:ANT,:SUBANT]
    surfaceregions = ["NATL","ANT","SUBANT"]
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
            # x = source_water_solution(surfaceregions,
            #     years,
            #     statevariables);
            
            x = source_water_solution(surfaceregions, years);

            @test fill(2.0,dims(x),:VectorArray) isa VectorArray
            @test ones(dims(x),:VectorArray) isa VectorArray
            @test randn(dims(x),:VectorArray) isa VectorArray

            ### slicing/broacasting
            #x[Ti=At(1990),:] # currently fails
            #x[Ti=At(1990)] # currently fails
            @test x[At(1990),:] isa VectorArray
            v = deepcopy(x)
            v[At(1990),:] = v[At(1990),:] .+ 1.0 
            @test sum(v-x) == length(surfaceregions)

            # slice the other way
            @test x[:,At("NATL")] isa VectorArray
            v = deepcopy(x)
            v[:,At("NATL")] = v[:,At("NATL")] .+ 1.0 
            @test sum(v-x) == length(years)
            
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

            λ, V = eigen(S)
            F = eigen(S)

            Λ = Diagonal(λ)
            #G = real(V * Λ / V) # also works
            G = V * Λ / V
            @test isapprox(Matrix(S), Matrix(G), atol = 1e-8)

            # check matrix exponential
            @test endomorphic(S)
            @test exp(S) isa MatrixArray # watch out for overflow!
        end
    end
end #"DimensionalData"
