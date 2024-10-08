#using Revise
using AlgebraicArrays
using Test
using ArraysOfArrays

@testset "AlgebraicArrays.jl" begin

    @testset "constructors" begin

        rsize = (2,3)

        # investigator makes a field with physical dimensions
        a = randn(rsize)

        # can immediately save it as a VectorArray for future calculations
        b = VectorArray(a)

        # internal algorithms must be able to turn into a vector, then bring it back to VectorArray
        c = AlgebraicArray(vec(a), rsize)
        @test a == c    

        # # make an array of arrays
        rsize = (1,2)
        dsize = (2,1)
        D = randn_MatrixArray(rsize,dsize)

        # internal algorithms must be able to turn into a matrix, then bring it back to a `MatrixArray`
        # turn a MatrixArray back into an array of arrays
        E = AlgebraicArray(Matrix(D),rsize,dsize)
        @test D == E 

        @testset "*,+,-,/,\\ and all that" begin

            rsize = (3,4)
            dsize = (2,3)

            q = VectorArray(randn(dsize))
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
            u = randn_VectorArray(usize)
            @test q * transpose(u) isa MatrixArray

            # another way to make a MatrixArray
            P = randn_MatrixArray(rsize,dsize)
    
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

            S = randn_MatrixArray(rsize,dsize)
            R = randn_MatrixArray(rsize,dsize)
            Q = R * S
            @test isapprox(Matrix(R \ Q), Matrix(S), atol = 1e-8)

            # # square matrices, matrix matrix right divide
            @test isapprox(Matrix(Q / S), Matrix(R), atol = 1e-8)

        end

        @testset "eigenstructure" begin

            rsize = (1,3)
            dsize = (1,3)
            x = VectorArray(randn(rsize))
            S = randn_MatrixArray(rsize,dsize)

            vals, vecs = eigen(S)
            F = eigen(S)

            Diagonal(vals)
            @test isapprox(Matrix(F), Matrix(S), atol= 1e-8)
            #println(typeof(real(AbstractMatrix(F))))
            #Matrix(F)
            # Sx_eigen = V * D / V
            # @test isapprox(Sx, Sx_eigen, atol = 1e-8)

            # # check matrix exponential
            # exp(Sx) # watch out for overflow!
        end

    end



    # rowdims = (3,3)
    # coldims = (3,3)
    # alldims = Tuple(vcat([i for i in rowdims],[j for j in coldims]))

    # E = nestedview(randn(alldims),length(coldims))
    # E[2,2][2,2]


    # F = E * E
    # F[2,2]

    # DT = transpose(D)
    # Ddagger = adjoint(D)


    # G = MatrixArray(Matrix(E))
    # G[2,2]
    # H = G * G
    # typeof(G)

    # J = MatrixArray(Matrix(G),(3,3),(3,3))

    # nested_array(A::AbstdractMatrix,rangedims,domaindims)

    #     # extra step for type stability
    #     Q1 = reshape(A[:,1],size(rangedims))
    #     P1 = DimArray(Q1, rangedims)

    #     P = Array{typeof(P1)}(undef,size(domaindims))
    #     for j in eachindex(P)
    #         Q = reshape(A[:,j],size(rangedims))
    #         P[j] = DimArray(Q, rangedims)
    #     end
    #     return DimArray(P, domaindims)
    # end

    #include("test_DimensionalData.jl")

end
