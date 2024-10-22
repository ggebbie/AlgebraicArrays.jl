@testset "unitful" begin

    using Unitful
    
    @testset "constructors" begin
        unitlist = [u"kg", u"K", u"m"]
        rsize = (2,3)
        
        # investigator makes a field with physical dimensions

        # not uniform
        nu = randn(rsize).*rand(unitlist,rsize)
        @test !isa(nu,Matrix{Quantity{T,S,V}} where {T,S,V})
        
        a = randn(rsize)*rand(unitlist) # uniform
        @test a isa Matrix{Quantity{T,S,V}} where {T,S,V}
        
        # can immediately save it as a VectorArray for future calculations
        b = VectorArray(a)

        # internal algorithms must be able to turn into a vector, then bring it back to VectorArray
        c = AlgebraicArray(vec(a), rsize)
        @test a == c    

        # # make an array of arrays
        rsize = (1,2)
        dsize = (2,1)
        D = randn(rsize,dsize, :MatrixArray)*rand(unitlist)

        # internal algorithms must be able to turn into a matrix, then bring it back to a `MatrixArray`
        # turn a MatrixArray back into an array of arrays
        E = AlgebraicArray(Matrix(D),rsize,dsize)
        @test D == E 

        @testset "*,+,-,/,\\ and all that" begin

            rsize = (3,4)
            dsize = (2,3)

            # uniform matrix for inner product
            q = randn(dsize, :VectorArray)*u"J"
            qT = transpose(q)
             # same type than q, but type instability in code
            qTT = transpose(qT)
            @test q == qTT

            # inner product
            @test ustrip(qT * q) ≥ 0

            # dot product not defined
            #@test q ⋅ q ≥ 0 

            # symmetric outer product
            @test q * qT isa MatrixArray

            # asymmetric outer product
            usize = (1,2)
            u = randn(usize, :VectorArray)*u"kg"
            @test q * transpose(u) isa MatrixArray

            # another way to make a MatrixArray
            P = randn(rsize,dsize,:MatrixArray) * rand(unitlist)
    
            # # multiplication of a MatrixArray and a VectorArray gives a VectorArray
            @test (P*q) isa VectorArray

            # # matrix-matrix multiplication
            PT = transpose(P)
            @test P * PT isa MatrixArray
            @test P == transpose(PT)

            P★ = adjoint(P)
            @test P * P★ isa MatrixArray

            r = P * q
            @test isapprox(vec(P \ r), vec(q), atol = 1e-8*unit(first(q))) # sometimes failed w/o `vec`

            # square matrices
            rsize = (2,3)
            dsize = (2,3)

            S = randn(rsize,dsize,:MatrixArray) * u"m"
            R = randn(rsize,dsize,:MatrixArray) * u"K"
            Q = R * S
            @test isapprox(Matrix(R \ Q), Matrix(S), atol = 1e-8*unit(first(first(S))))

            # # square matrices, matrix matrix right divide
            @test isapprox(Matrix(Q / S), Matrix(R), atol = 1e-8*unit(first(first(R)))) # fails due to Quantity(F64) error

        end

        @testset "eigenstructure" begin

            rsize = (1,3)
            dsize = (1,3)
            S = randn(rsize,dsize,:MatrixArray)*rand(unitlist)

            @test S isa MatrixArray{T1,N,M,Matrix{Quantity{T2,S,V}}} where {T1,T2,N,M,S,V}
            
            vals, vecs = eigen(S)
            F = eigen(S)
            @test isapprox(Matrix(F), Matrix(S), atol= 1e-8*unit(first(first(S))))

        end

    end

end
