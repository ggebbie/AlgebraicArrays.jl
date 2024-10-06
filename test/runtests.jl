using Revise
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
        c = VectorArray(vec(a), rsize)
        @test a == c    

        # # make an array of arrays
        rsize = (1,2)
        dsize = (2,1)
        alldims = Tuple(vcat([i for i in rsize],[j for j in dsize]))

        # investigator/algorithm makes a field of fields with physical dimensions
        C = Matrix(nestedview(randn(alldims),length(dsize)))
    
        # turn an array of arrays into a MatrixArray for future calculations
        D = MatrixArray(C)

        # internal algorithms must be able to turn into a matrix, then bring it back to a `MatrixArray`
        # turn a MatrixArray back into an array of arrays
        E = MatrixArray(Matrix(D),rsize,dsize)
        @test D == E 

        @testset "multiplication" begin

            rsize = (3,4)
            dsize = (2,3)

            q = VectorArray(randn(dsize))

            # # make an array of arrays
            alldims = Tuple(vcat([i for i in rsize],[j for j in dsize]))
            P = MatrixArray(Matrix(nestedview(randn(alldims),length(dsize))))
    
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


end
