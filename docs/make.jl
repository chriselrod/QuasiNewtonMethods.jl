using Documenter, QuasiNewtonMethods

makedocs(;
    modules=[QuasiNewtonMethods],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/QuasiNewtonMethods.jl/blob/{commit}{path}#L{line}",
    sitename="QuasiNewtonMethods.jl",
    authors="Chris Elrod <elrodc@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/chriselrod/QuasiNewtonMethods.jl",
)
