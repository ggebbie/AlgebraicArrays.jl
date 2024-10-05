using Revise
using AlgebraicArrays
using Test

@testset "AlgebraicArrays.jl" begin
    # Write your tests here.

    rsize = (2,3)
    a = randn(rsize)
    b = VectorArray(a)
    c = VectorArray(b, rsize)
    @test a == c    

    # # make an array of arrays
    # rowdims = (4,4)
    # coldims = (5,5)
    # alldims = Tuple(vcat([i for i in rowdims],[j for j in coldims]))

    # C = nestedview(randn(alldims),length(coldims))
    # C[5,5][4,4]

    # # turn an array of arrays into a MatrixArray
    # D = MatrixArray(C)
    # D = MatrixArray(Matrix(C))

    # # turn a MatrixArray into an algebraic array
    # Matrix(D)

    # # multiplication of a MatrixArray and a VectorArray gives a VectorArray
    # q = D*b

    # rsize = (4,4)
    # dsize = (5,5)
    # MatrixArray(Matrix(D),rsize, dsize) # doesn't work 

    # # matrix-matrix multiplication
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
