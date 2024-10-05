using AlgebraicArrays
using Documenter

DocMeta.setdocmeta!(AlgebraicArrays, :DocTestSetup, :(using AlgebraicArrays); recursive=true)

makedocs(;
    modules=[AlgebraicArrays],
    authors="G Jake Gebbie <ggebbie@whoi.edu>",
    sitename="AlgebraicArrays.jl",
    format=Documenter.HTML(;
        canonical="https://ggebbie.github.io/AlgebraicArrays.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ggebbie/AlgebraicArrays.jl",
    devbranch="main",
)
