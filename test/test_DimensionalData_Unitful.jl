@testset "Dimensional Data Unitful" begin

    # opinion: Unitful is really only convenient for uniform matrices
    # restrict to uniform matrices
    
    # fixed parameters already set in DimensionalData section
    # @dim YearCE "years Common Era"
    # @dim SurfaceRegion "surface location"
    # @dim InteriorLocation "interior location"
    # @dim StateVariable "state variable"
    #surfaceregions = [:NATL,:ANT,:SUBANT]
    surfaceregions = ["NATL","ANT","SUBANT"]
    N = length(surfaceregions)
    years = (1990:1993)
    statevariables = [:θ, :δ¹⁸O] 
    M = 5 # Interior Locations with obs

    function source_water_solution_with_uniform_units(surfaceregions, years, statevar)
        yr = u"yr"
        K = u"K"
        permil = u"permille"
        m = length(years)
        n = length(surfaceregions)
        mat = cat(randn(m, n, 1), randn(m, n, 1); dims = 3)K
        x = VectorArray(DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar))))
        return x
    end

    @testset "AlgebraicArrays + DimensionalData + Unitful" begin

        MatrixDimArray = MatrixArray{T, M, N, R} where {M, T, N, R<:AbstractDimArray{T, M}}
        VectorDimArray = VectorArray{T, N, A} where {T, N, A <: DimensionalData.AbstractDimArray}

        @testset "with units" begin
            x = source_water_solution_with_uniform_units(surfaceregions,
                years,
                statevariables)

            @test x isa VectorDimArray

            K = unit(first(x))
            @test fill(2.0,dims(x),:VectorArray)K isa VectorDimArray
            @test ones(dims(x),:VectorArray)K isa VectorDimArray
            @test randn(dims(x),:VectorArray)K isa VectorDimArray

            @testset "inner and outer products" begin
                xT = transpose(x)
                @test xT isa MatrixDimArray

                xTT = transpose(xT)
                @test x == xTT

                @test xT * x ≥ 0K^2
                @test x ⋅ x ≥ 0K^2
                @test isapprox(xT * x, x ⋅ x)
            end
            
            @testset "slicing and broadcasting" begin
                @test x[Ti=At(1990)] isa VectorDimArray

                getindex(x,At(1990),:,:)
                @test x[At(1990),:,:] isa VectorDimArray
                @test 2*x[At(1990),:,:] isa VectorDimArray
                @test 2.0*x[At(1990),:,:] isa VectorDimArray
                @test 2.0.*x[At(1990),:,:] isa VectorDimArray # fails for non-uniform
                                
                v = deepcopy(x)
                @test v[:,:,At(:θ)] .+ 1.0K isa VectorDimArray # fails for non-uniform vector
                #v[At(1990),:,:] .= 2.0 * v[At(1990),:,:]  # unable to check bounds
                v[1,:,:] .= 2.0 .* v[1,:,:] # workaround  
                v[1,:,:] .= 2.0 * v[1,:,:]  # workaround
                parent(v)[At(1990),:,:] .= 2.0 * v[At(1990),:,:] # workaround

                v = deepcopy(x)
                parent(v)[At(1990),:,:] .+= 1.0K
                @test isapprox(ustrip.(sum(v-x)),length(surfaceregions)*length(statevariables)) # incompatible with non-uniform units

                v = deepcopy(x)
                # v[At(1990),:,:] .*=  2.0K # fails due to check bounds
                v[1,:,:] .*=  2.0 # succeeds
                parent(v)[At(1990),:,:] .*=  2.0 # another workaround
            
                # slice the other way
                @test x[:,At("NATL"),:] isa VectorDimArray
                v = deepcopy(x)
                #v[:,At("NATL"),:] .* 2.0 # fails due to dims mismatch
                #v[:,1,:] .* 2.0 # fails due to dims mismatch
                v[:,At("NATL"),:] * 2.0 # ok
                #v[:,At("NATL"),:] .= v[:,At("NATL"),:] * 2.0 # fails due to check bounds
                parent(v)[:,At("NATL"),:] .= v[:,At("NATL"),:] * 2.0 # workaround
                v[:,1,:] .= v[:,1,:] * 2.0 # another workaround

                #v[SurfaceRegion=At("NATL")] .= 2.0 * v[SurfaceRegion=At("NATL")]  # fails, no dotview
                parent(v)[SurfaceRegion=At("NATL")] .= 2.0 * v[SurfaceRegion=At("NATL")]  # workaround
                v[SurfaceRegion=At("NATL"),Ti=At(1990)]
                #@test isapprox(sum(v-x), length(years)) # not compatible with non-uniform units

                # dot multiply
                @test v .* v isa VectorDimArray # fails for non-uniform
                @test transpose(v) * v isa Number # fails for non-uniform
                @test v ⋅ v isa Number # fails for non-uniform
                @test isapprox(transpose(v) * v, v ⋅ v)
            end 
            
            # test that these vectors;matrices can be used in algebraic expressions
            y = vec(x)
            z = AlgebraicArray(y, rangedims(x))
            @test x == z

            # make the diagonal elements
            w = ones(dims(x), :VectorArray)*u"J" # uniform matrix, weak test
            D = Diagonal(w)
            DT = transpose(D)
            DTT = transpose(DT)
            @test D == DT
            @test D == DTT

            R = AlgebraicArray(randn(length(x),length(x)),
                rangedims(x), rangedims(x))/K
            RT = transpose(R)
            RTT = transpose(RT)
            @test R == RTT
            @test R ≠ RT
                        
            q = R * x
            @test q isa VectorArray{T,N,DA} where T where N where DA <: DimensionalData.AbstractDimArray
            @test q isa VectorDimArray
        
            y = R \ q
            @test isapprox(x, y)

            S = AlgebraicArray(randn(length(x),length(x)),
                rangedims(x), rangedims(x))u"J"    
            Q = R * S
            U = R \ Q 
            @test isapprox(Matrix(U), Matrix(S))

            # square matrices, matrix matrix right divide
            @test isapprox(Matrix(Q / S), Matrix(R))

            @testset "matrix slicing" begin
                @test R[1] isa VectorDimArray
                @test R[2,1,1] isa VectorDimArray
                @test R[1:2,1,1] isa MatrixDimArray
                @test R[1:2] isa MatrixDimArray

                R2 = deepcopy(R)
                R2[2,1,1] .+= 1.0/K 
                @test all(isapprox.(sum(R2-R), 1.0/K))

                # iteration uses CartesianIndices not linear indices, would need to set `iterate` function 
                # @test eachindex(D) == Base.OneTo(prod(size(b)))

                @test R[2,1,2][1,1,1] isa Number
                @test rowvector(R,1,1,2) isa MatrixDimArray

                @test R[At(1990),At("NATL"),At(:θ)] isa VectorDimArray
                @test R[At(1990:1991),At("NATL"),:] isa MatrixDimArray
                @test R[At(1990:1991),:,:] isa MatrixDimArray
                #@test R[Ti=At(1990:1991)] isa MatrixDimArray # fails, invalid index

                R2 = deepcopy(R)
                #R2[At(1990),At("NATL"),At(:θ)] .*= 2.0 #fails, cannot check bounds
                parent(R2)[At(1990),At("NATL"),At(:θ)] .+= 1.0/K #workaround 
                @test all(isapprox.(sum(R2-R), 1.0/K))

                # iteration uses CartesianIndices not linear indices, would need to set `iterate` function 
                # @test eachindex(D) == Base.OneTo(prod(size(b)))

                @test R[At(1990),At("NATL"),At(:θ)][At(1990),At("NATL"),At(:θ)] isa Number
                #@test R[:][At(1990),At("NATL")] isa VectorDimArray # fails, upstream DD issue?
                @test rowvector(R,At(1990),At("NATL"),At(:θ)) isa MatrixDimArray

                # setindex!
                R[At(1990),At("NATL"),At(:θ)][At(1990),At("NATL"),At(:θ)] = 0.0/K
                # set columns to be equal
                #R[At(1990),At("NATL"),At(:θ)] .= R[At(1990),At("ANT"),At(:θ)]  # fails, check bounds
                parent(R)[At(1990),At("NATL"),At(:θ)] .= R[At(1990),At("ANT"),At(:θ)]  # workaround
                @test all(isapprox.(transpose(Matrix(R)[1,:]), Matrix(rowvector(R,1))))

            end

        end

        @testset "eigenstructure" begin 
            x = source_water_solution_with_uniform_units(surfaceregions,
                years,
                statevariables)

            S = AlgebraicArray(randn(length(x),length(x)),
                rangedims(x), rangedims(x))u"K"    

            λ, V = eigen(S)
            F = eigen(S)
            @test isapprox(Matrix(F), Matrix(S))
            
            Λ = Diagonal(λ)
            #G = real(V * Λ / V) # also works
            G = V * Λ / V
            @test isapprox(Matrix(S), Matrix(G))

            # matrix exponential: not valid for uniform matrices (except nondimensional one)
            # @test AlgebraicArrays.endomorphic(S)
            # @test exp(S) isa MatrixDimArray # watch out for overflow!
        end
    end
end #"DimensionalData"
