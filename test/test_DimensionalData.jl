@testset "Dimensional Data" begin

    #surfaceregions = [:NATL,:ANT,:SUBANT]
    surfaceregions = ["NATL","ANT","SUBANT"]
    N = length(surfaceregions)
    years = (1990:1993)
    statevariables = [:θ, :δ¹⁸O] 
    M = 5 # Interior Locations with obs

    function source_water_solution(surfaceregions,years)
        m = length(years)
        n = length(surfaceregions)
        da = DimArray(randn(m,n),(Ti(years),SurfaceRegion(surfaceregions)))

        # wrap it with an AlgebraicArray
        return AlgebraicArray(da, (size(da),))
        # return VectorArray(DimArray(randn(m,n),(Ti(years),SurfaceRegion(surfaceregions))))
    end

    function source_water_solution(surfaceregions, years, statevar)
        m = length(years)
        n = length(surfaceregions)
        mat = cat(randn(m, n, 1), randn(m, n, 1); dims = 3)
        da = DimArray(mat, (Ti(years), SurfaceRegion(surfaceregions), StateVariable(statevar)))
        x = VectorArray(da, (size(da),))
        return x
    end

    @testset "AlgebraicArrays + DimensionalData.jl" begin

        MatrixDimArray = MatrixArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}
        VectorDimArray = VectorArray{T, N, A} where {T, N, A<:AbstractDimArray{T, N}}

        @testset "no units" begin
            # x = source_water_solution(surfaceregions,
            #     years,
            #     statevariables);

            x = source_water_solution(surfaceregions, years)

            @test x isa VectorDimArray
            @test fill(2.0,dims(x),(size(dims(x)),)) isa VectorDimArray
            @test ones(dims(x), (size(dims(x)),)) isa VectorDimArray
            @test rand(dims(x), (size(dims(x)),)) isa VectorDimArray

            @testset "inner and outer products" begin
                xT = transpose(x) 
                @test xT isa MatrixDimArray 

                xTT = transpose(xT)
                # @test x == xTT # fails because singleton dimension not dropped
                @test x[3] == xTT[3]

                # inner product
                # @test xT * x ≥ 0 # fails because return 1-vector
                @test first(xT * x) ≥ 0 
                @test x ⋅ x ≥ 0
                @test isapprox(first(xT * x), x ⋅ x)
            end
            
            @testset "slicing and broadcasting" begin
                # @test x[Ti=At(1990)] isa VectorDimArray # fails
                # @test x[(Ti=At(1990))] isa VectorDimArray # fails
                # @test parent(x)[Ti=At(1990)] isa VectorDimArray # fails
                @test x[At(1990),:] isa VectorDimArray
                @test x[(At(1990),:)] isa VectorDimArray

                getindex(x,At(1990),:)
                @test x[At(1990),:] isa VectorDimArray
                v = deepcopy(x)
                v[At(1990),:] = v[At(1990),:] .+ 1.0 
                @test isapprox(sum(v-x),length(surfaceregions))

                v = deepcopy(x)
                #v[At(1990),:] .+=  1.0 #fails
                # v[1,:] .+=  1.0 # succeeded before, now fails
                # @test isapprox(sum(v-x), length(surfaceregions))
            
                # slice the other way
                @test x[:,At("NATL")] isa VectorArray
                v = deepcopy(x)
                v[:,At("NATL")] = v[:,At("NATL")] .+ 1.0 
                #v[:,At("NATL")] .+= 1.0  #fails
                #v[SurfaceRegion=At("NATL")] = v[SurfaceRegion=At("NATL")] .+ 1.0 # fails, not recommended
                @test isapprox(sum(v-x), length(years))

                # dot multiply
                # @test v .* v isa VectorDimArray # fails
                @test v .* v isa VectorArray # but is close
                
            end 
            
            # test that these vectors;matrices can be used in algebraic expressions
            # testing a constructor that no longer exists?
            # y = vec(x)
            # z = AlgebraicArray(y, dims(parent(x)))
            # @test x == z

            # make the diagonal elements
            w = ones(dims(x), (size(dims(x)),))
            D = Diagonal(w)
            DT = transpose(D)
            DTT = transpose(DT)
            @test D == DT
            @test D == DTT

            Rda = rand((rangedims(x)..., rangedims(x)...)) - rand((rangedims(x)..., rangedims(x)...))
            R = AlgebraicArray(Rda, (size(rangedims(x)), size(rangedims(x))))
            RT = transpose(R)
            RTT = transpose(RT)
            @test R == RTT
            @test R ≠ RT
            @test similar(R) isa MatrixDimArray

            @testset "matrix construction" begin

                funks = [:rand,:zeros,:ones]
                for fnk in funks
                    #J = randn(rsize,rsize,:MatrixArray)
                    rsize = dims(x)
                    #J = @eval $fnk((1,2),(1,2),:MatrixArray)
                    J = @eval $fnk(($rsize..., $rsize...),(size($rsize),size($rsize)))
                    @test endomorphic(J) 
                    @test diag(J) isa VectorArray
                    id = rand(1:size(J,1))
                    @test diag(J)[id] == J[id,id]
                end

                #fill
                J =  fill(1,(dims(x)..., dims(x)...),(size(dims(x)),size(dims(x))))
                @test endomorphic(J) 
                @test diag(J) isa VectorArray
                id = rand(1:size(J,1))
                @test diag(J)[id] == J[id,id]
            end

            @testset "matrix slicing" begin
                
                @test R[(1,1),(1,1)] isa Number 
                @test R[(1,2),(1,1)] isa Number
                @test R[(1,1:2),(1,1)] isa VectorDimArray
                @test R[(:,1:2),(1:2,:)] isa MatrixDimArray
                @test R[(1,2),(:,:)] isa MatrixDimArray # row vector but Julia returns a 1 x N matrix

                R2 = deepcopy(R)
                # R2[(:,:),(1,2)] .+= 1.0 # fails
                parent(R2)[:,:,1,2] .+= 1.0 # workaround
                @test all(isapprox.(sum(R2-R), prod(size(rangedims(R2)))))

                # iteration uses CartesianIndices not linear indices, would need to set `iterate` function 
                # @test eachindex(D) == Base.OneTo(prod(size(b)))

                @test R[(2,1),(1,1)] isa Number
                @test R[(:,:),(1,1)] isa VectorDimArray # actually this is not the same as rowvector and is incorrect
                @test R[(1,1),(:,:)] isa MatrixDimArray

                @test R[(:,:),(At(1990),At("NATL"))] isa VectorDimArray
                @test R[(:,:),(At(1990:1991),At("NATL"))] isa MatrixDimArray
                @test R[(:,:),(At(1990:1991),:)] isa MatrixDimArray

                R2 = deepcopy(R)
                #R2[(:,:),(At(1990),At("NATL"))] .+= 1.0 #fails
                parent(R2)[At(1990),At("NATL"),:,:] .+= 1.0 #workaround 
                @test all(isapprox.(sum(R2-R), prod(size(domaindims(R2)))))

                # iteration uses CartesianIndices not linear indices, would need to set `iterate` function 
                @test eachindex(R2) isa CartesianIndices
                # @test eachindex(D) == Base.OneTo(prod(size()))

                @test R2[(At(1990),At("NATL")),(At(1990),At("NATL"))] isa Number
                @test R2[(At(1990),At("NATL")),(:,:)] isa MatrixDimArray 

                # setindex!
                R[(At(1990),At("NATL")),(At(1990),At("NATL"))] = 0.0

                parent(R)[At(1990),At("NATL"),:,:] .= 0.0 # workaround
                #                 R[(At(1990),At("NATL")),(:,:)] .= 0.0 # fails

                # set columns to be equal
                #parent(R)[At(1990),At("NATL")] .= R[At(1990),At("AABW")]  # fails, but works with numerical indices
                # parent(R)[At(1990),At("NATL"),:,:] .= parent(R)[At(1990),At("AABW"),:,:]  # still fails
                parent(R)[1,1,:,:] .= parent(R)[2,1,:,:]  # workaround

                @test all(isapprox.(transpose(Matrix(R)[1,:]), Matrix(R[(1,1),(:,:)])))

            end
            
            q = R * x
            @test q isa VectorArray{T,N,DA} where T where N where DA <: DimensionalData.AbstractDimArray
            @test q isa VectorDimArray
        
            y = R \ q
            @test isapprox(x, y, atol = 1e-8)

            S = rand((dims(x)..., dims(x)...), (size(dims(x)),size(dims(x)))) 
            @test S isa MatrixDimArray

            Q = R * S
            U = R \ Q 
            @test isapprox(Matrix(U), Matrix(S), atol = 1e-8)

            # square matrices, matrix matrix right divide
            @test isapprox(Matrix(Q / S), Matrix(R), atol = 1e-8)

            # non-square multiplication
            G = S[(At(1990),:),(:,:)]
            G * S
            # rsiqze = (2,3)
            # dsize = (1,3)
            # G = randn(rsize,dsize,:MatrixArray) 
            # H = randn(dsize,rsize,:MatrixArray) 
            # Matrix(G * H)
            
        end

        @testset "eigenstructure" begin 
            x = source_water_solution(surfaceregions,
                years,
                statevariables)

            S = AlgebraicArray(rand(length(x),length(x)),
                rangedims(x), rangedims(x))    

            λ, V = eigen(S)
            F = eigen(S)
            @test isapprox(Matrix(F), Matrix(S), atol= 1e-8)
            
            Λ = Diagonal(λ)
            #G = real(V * Λ / V) # also works
            G = V * Λ / V
            @test isapprox(Matrix(S), Matrix(G), atol = 1e-8)

            # check matrix exponential
            @test AlgebraicArrays.endomorphic(S)
            @test exp(S) isa MatrixDimArray # watch out for overflow!
        end
    end
end #"DimensionalData"
