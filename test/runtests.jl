using Revise
using AlgebraicArrays
using Test
using ArraysOfArrays
using DimensionalData
using DimensionalData:@dim
using Unitful

@testset "AlgebraicArrays.jl" begin

    @testset "constructors" begin

        rsize = (2,3)

        @test fill(2.0,rsize,:VectorArray) isa VectorArray
        @test ones(rsize,:VectorArray) isa VectorArray
        @test randn(rsize,:VectorArray) isa VectorArray
        
        # investigator makes a field with physical dimensions
        a = randn(rsize)

        # can immediately save it as a VectorArray for future calculations
        b = VectorArray(a)

        ### slicing + broadcasting
        @test b[1,:] isa VectorArray
        v = deepcopy(b)
        v[1,:] = v[1,:] .+ 1.0 
        @test sum(v-b) == rsize[2]
        
        # internal algorithms must be able to turn into a vector, then bring it back to VectorArray
        c = AlgebraicArray(vec(a), rsize)
        @test a == c    

        # test `similar`
        @test similar(c) isa VectorArray

        # custom broadcasting
        @test all(abs.(c) .> 0)
        
        # # make an array of arrays
        rsize = (1,2)
        dsize = (2,1)
        D = randn(rsize,dsize,:MatrixArray) #randn_MatrixArray(rsize,dsize)

        # internal algorithms must be able to turn into a matrix, then bring it back to a `MatrixArray`
        # turn a MatrixArray back into an array of arrays
        E = AlgebraicArray(Matrix(D),rsize,dsize)
        @test D == E 

        # not possible to broadcast to nested array
        F = real(D)
        
        @testset "*,+,-,/,\\ and all that" begin

            rsize = (3,4)
            dsize = (2,3)

            q = randn(dsize,:VectorArray) #VectorArray(randn(dsize))
            qT = transpose(q)
             # same type than q, but type instability in code
            qTT = transpose(qT)
            @test q == qTT

            # inner product
            @test qT * q ≥ 0

            # dot product is not correct
            #@test q ⋅ q ≥ 0 

            # symmetric outer product
            @test q * qT isa MatrixArray

            # asymmetric outer product
            usize = (1,2)
            u = randn(usize,:VectorArray) # formerly randn_VectorArray(usize)
            @test q * transpose(u) isa MatrixArray

            # another way to make a MatrixArray
            P = randn(rsize,dsize,:MatrixArray) #randn_MatrixArray(rsize,dsize)
    
            # # multiplication of a MatrixArray and a VectorArray gives a VectorArray
            @test (P*q) isa VectorArray

            # # matrix-matrix multiplication
            PT = transpose(P)
            @test P * PT isa MatrixArray
            @test P == transpose(PT)

            P★ = adjoint(P)
            @test P * P★ isa MatrixArray

            r = P * q
            @test isapprox(vec(P \ r), vec(q), atol = 1e-8) # sometimes failed w/o `vec`

            # square matrices
            rsize = (2,3)
            dsize = (2,3)

            S = randn(rsize,dsize,:MatrixArray) #randn_MatrixArray(rsize,dsize) 
            R = randn(rsize,dsize,:MatrixArray) #randn_MatrixArray(rsize,dsize)
            Q = R * S
            @test isapprox(Matrix(R \ Q), Matrix(S), atol = 1e-8)

            # # square matrices, matrix matrix right divide
            @test isapprox(Matrix(Q / S), Matrix(R), atol = 1e-8)

        end

        @testset "eigenstructure" begin

            rsize = (2,3)
            dsize = (2,3)
            x = randn(rsize,:VectorArray) #VectorArray(randn(rsize))
            S = randn(rsize,dsize,:MatrixArray) #randn_MatrixArray(rsize,dsize)

            vals, vecs = eigen(S)
            F = eigen(S)

            Diagonal(vals)
            @test isapprox(Matrix(F), Matrix(S), atol= 1e-8)

            # # check matrix exponential
            @test exp(S) isa MatrixArray # watch out for overflow!
        end
    end

    include("test_DimensionalData.jl")
    include("test_unitful.jl")
    
end
