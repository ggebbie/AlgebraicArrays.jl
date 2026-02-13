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

    # what if dims don't match
    c = AlgebraicArray(vec(a), vsize)

    @test b == c

    @testset "vector broadcasting and slicing" begin
        # use physical indices
        @test b[(1,:)] isa VectorArray
        @test b[(1:2,:)] isa VectorArray

        # currently failing
        # @test b[:,2:end] isa VectorArray # end keyword not correct
        @test b[:,2:3] isa VectorArray # end keyword not correct

        v = deepcopy(b)

        # v[(1,:)] .+= 1.0         # currently failing
        parent(v)[1,:] .+= 1.0         # workaround
        
        # v[1,:] .+= 1.0         # currently failing
        parent(v)[1,:] .+= 1.0         # workaround
        
        # v[1] .+= 1.0     # currently failing
        #parent(v)[1] .+= 1.0     # currently failing with 
        # ERROR: MethodError: no method matching copyto!(::Float64, ::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{…}, Tuple{}, typeof(+), Tuple{…}})
        # The function `copyto!` exists, but no method is defined for this combination of argument types.

        v = deepcopy(b)
        v[1,:] = v[1,:] .+ 1.0 # works
        #v[(1,:)] = v[(1,:)] .+ 1.0 # fails
        v[1] = v[1] .+ 1.0 # works
        @test isapprox(sum(v-b), dsize[2] + 1) # change to factor 3 when above error fixed

        # iteration
        @test eachindex(b) == Base.OneTo(prod(size(b)))
    end
        
    # internal algorithms must be able to turn into a vector, then bring it back to VectorArray
    # c = AlgebraicArray(a, size) 
    # @test a == c    

    # test `similar`
    @test similar(b) isa VectorArray

    # custom broadcasting
    @test all(abs.(b) .> 0) # careful that it doesn't change type
        
    rsize = (1,2)
    dsize = (2,1)
    msize = (dsize, rsize)
    mdata = fill(2.0, AlgebraicArrays.unwrap(msize))
    C = AlgebraicArray(mdata, msize) 
    D = fill(2.0, msize)

    @test C == D
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

    #fil
    J = fill(1, ((2,1),(2,1)))
    @test endomorphic(J) 
    @test diag(J) isa VectorArray
    id = rand(1:size(J,1))
    @test diag(J)[id] == J[id,id]

    # internal algorithms must be able to turn into a matrix, then bring it back to a `MatrixArray`
    # turn a MatrixArray back into an array of arrays: still true?
    # E = AlgebraicArray(Matrix(D),rsize,dsize)
    # @test D == E 
    # @test similar(D) isa MatrixArray
        
    @testset "matrix slicing" begin

        # will never slice the algebraic array.
        # only slice the dimensional array
        @test D[(1,1),(1,1)] isa Number
        @test D[(2,1),(1,1)] isa Number
        @test D[(1:2,1),(1,1)] isa VectorArray
        @test D[(1:2,:),(:,1:2)] isa MatrixArray
        @test D[(2,1),(:,:)] isa MatrixArray # row vector but Julia returns a 1 x N matrix

        # iteration uses CartesianIndices not linear indices, would need to set `iterate` function 
        @test eachindex(D) isa CartesianIndices
        # @test eachindex(D) == Base.OneTo(prod(size(D)))

        D2 = deepcopy(D)

        # setindex!
        # D2[(:,:),(1,2)] .+= 1.0 # currently failing
        parent(D2)[:,:,1,2] .+= 1.0 # workaround
        @test all(isapprox.(sum(D2-D), prod(domaindims(D))))

        # D[(2,1),(1,1)] = 0.0 # failing
        parent(D)[2,1,1,1] = 0.0 # workaround

        # set columns to be equal
        # D[(:,:),(1,2)] .= D[(:,:),(1,1)] # failing
        parent(D)[:,:,1,2] .= parent(D)[:,:,1,1] # workaround

        # set rows to be equal
        # D[(2,1)(:,:)] .= D[(1,1),(:,:)] # failing
        parent(D)[2,1,:,:] .= parent(D)[1,1,:,:] # workaround

        D = randn(msize)
        Drow1 = Matrix(D[(1,1),(:,:)])
        Drow2 = transpose(Matrix(D)[1,:])
        @test isapprox(Drow1, Drow2)
    end
        
    # now possible to broadcast 
    F = real.(D)
    @test typeof(F) == typeof(D)
    
    @testset "*,+,-,/,\\ and all that" begin

        rsize = (3,4)
        dsize = (2,3)
        msize = (rsize, dsize)

        q = randn((dsize,))
        qT = transpose(q)
        @test q[2] == qT[1,2]

        # same type than q, but type instability in code
        qTT = transpose(qT)

        itest = rand(eachindex(q))
        @test q[itest] == qTT[itest]

        # inner product
        @test first(qT * q) ≥ 0 # workaround
        # @test qT * q ≥ 0 # returns Vector, should be Number

        @test q ⋅ q ≥ 0 

        @test isapprox(first(qT * q), q ⋅ q) # workaround
        # @test isapprox(qT * q, q ⋅ q) # fails

        # symmetric outer product
        @test q * qT isa MatrixArray

        # asymmetric outer product
        usize = (1,2)
        u = randn((usize,)) 
        @test q * transpose(u) isa MatrixArray

        # another way to make a MatrixArray
        P = randn(msize)
        @test rangedims(P) == rsize
        @test domaindims(P) == dsize
            
        # # multiplication of a MatrixArray and a VectorArray gives a VectorArray
        @test (P*q) isa VectorArray

        # # matrix-matrix multiplictation
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
