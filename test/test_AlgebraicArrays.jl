@testset "constructors" begin

    # size of a VectorArray
    rsize = (2,3)
    vsize = (rsize,) # tells it to make a vector
    @test fill(2.0,vsize) isa VectorArray  
    @test ones(vsize) isa VectorArray
    @test randn(vsize) isa VectorArray
    @test zeros(vsize) isa VectorArray 
    @test rand(vsize) isa VectorArray
        
    # investigator makes a field with physical dimensions
    a = randn(rsize)

    # can immediately save it as a VectorArray for future calculations
    b = AlgebraicArray(a, vsize)

    @testset "vector broadcasting and slicing" begin
        @test b[1,:] isa VectorArray
        @test b[1:2,:] isa VectorArray
        # @test b[:,2:end] isa VectorArray # end keyword not correct
        v = deepcopy(b)
        v[1,:] .+= 1.0 
        @test isapprox( sum(v-b), rsize[2])

        v = deepcopy(b)
        v[1,:] = v[1,:] .+ 1.0 
        @test isapprox(sum(v-b), rsize[2])

        # iteration
        @test eachindex(b) == Base.OneTo(prod(size(b)))
    end
        
    # internal algorithms must be able to turn into a vector, then bring it back to VectorArray
    # c = AlgebraicArray(a, size) 
    # @test a == c    

    # test `similar`
    @test similar(b) isa VectorArray

    # custom broadcasting
    @test all(abs.(b) .> 0)
        
    # # make an array of arrays
    rsize = (1,2)
    dsize = (2,1)
    msize = (dsize, rsize)
    mdata = fill(2.0, AlgebraicArrays.unwrap(msize))
    C = AlgebraicArray(mdata, msize) 
    D = randn(msize)

    @test !endomorphic(D)
    @test !(diag(D) isa VectorArray)

    @test rangedims(D) == rsize
    @test domaindims(D) == dsize

    funks = [:randn,:zeros,:ones]
    for fnk in funks
        rsize = (1,2)
        msize = (rsize, rsize)
        J = @eval $fnk($msize)
        @test endomorphic(J) 
        @test diag(J) isa VectorArray
        id = rand(1:size(J,1))
        @test diag(J)[id] == J[id,id]
    end

    #fill
    J = fill(1, msize)
    @test endomorphic(J) 
    @test diag(J) isa VectorArray
    id = rand(1:size(J,1))
    @test diag(J)[id] == J[id][id]

    # internal algorithms must be able to turn into a matrix, then bring it back to a `MatrixArray`
    # turn a MatrixArray back into an array of arrays: still true?
    # E = AlgebraicArray(Matrix(D),rsize,dsize)
    # @test D == E 
    # @test similar(D) isa MatrixArray
        
    @testset "matrix slicing" begin
        @test D[1] isa VectorArray
        @test D[2,1] isa VectorArray
        @test D[1:2,1] isa MatrixArray
        @test D[1:2] isa MatrixArray

        D2 = deepcopy(D)
        D2[2,1] .+= 1.0 
        @test all(isapprox.(sum(D2-D), 1.0))

        # iteration uses CartesianIndices not linear indices, would need to set `iterate` function 
        # @test eachindex(D) == Base.OneTo(prod(size(b)))

        @test D[2,1][1,1] isa Number
        @test rowvector(D,1,1) isa MatrixArray
        @test all(isapprox.(transpose(Matrix(D)[1,:]), Matrix(rowvector(D,1,1))))
            
        # setindex!
        D[2,1][1,1] = 0.0
        # set columns to be equal
        D[2,1] .= D[1,1]
        # set rows to be equal
        #rowvector(D,1) .= 0.0 # fails, use comprehension instead
            
    end
        
    # not possible to broadcast to nested array
    F = real(D)
        
    @testset "*,+,-,/,\\ and all that" begin

        rsize = (3,4)
        dsize = (2,3)
        msize = (dsize, rsize)
        
        q = randn(dsize,:VectorArray) #VectorArray(randn(dsize))
        qT = transpose(q)
        # same type than q, but type instability in code
        qTT = transpose(qT)
        @test q == qTT

        # inner product
        @test qT * q ≥ 0

        # dot product is not correct
        @test q ⋅ q ≥ 0 
        @test isapprox(qT * q, q ⋅ q)

        # symmetric outer product
        @test q * qT isa MatrixArray

        # asymmetric outer product
        usize = (1,2)
        u = randn(usize,:VectorArray) # formerly randn_VectorArray(usize)
        @test q * transpose(u) isa MatrixArray

        # another way to make a MatrixArray
        P = randn(rsize,dsize,:MatrixArray) #randn_MatrixArray(rsize,dsize)
        @test rangedims(P) == rsize
        @test domaindims(P) == dsize
            
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

        # non-square multiplication
        rsize = (2,3)
        dsize = (1,3)
        G = randn(rsize,dsize,:MatrixArray) 
        H = randn(dsize,rsize,:MatrixArray) 
        Matrix(G * H)
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
