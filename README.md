# QuasiNewtonMethods

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/QuasiNewtonMethods.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriselrod.github.io/QuasiNewtonMethods.jl/dev)
![CI](https://github.com/chriselrod/QuasiNewtonMethods.jl/workflows/CI/badge.svg)
![CI (Julia nightly)](https://github.com/chriselrod/QuasiNewtonMethods.jl/workflows/CI%20(Julia%20nightly)/badge.svg)
[![Codecov](https://codecov.io/gh/chriselrod/QuasiNewtonMethods.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/QuasiNewtonMethods.jl)



---


This library aims to be fast. It's intended use is for optimizing in statistical logdensity functions, in particular in conjunction with ProbabilityModels.jl and InplaceDHMC.jl (both libraries are still undergoing major development and are not yet usable). The API thus requires using `logdensity` functions, which `ProbabilityModels` will automatically define for a given model:
```julia
using QuasiNewtonMethods, StrideArrays
using Test

struct Rosenbrock end
function QuasiNewtonMethods.logdensity(::Rosenbrock, θ)
    s = zero(eltype(θ))
    N = length(θ) >> 1
    @inbounds @simd for i ∈ 1:N
        s -= 100(θ[i+N] - θ[i]^2)^2 + (1 - θ[i])^2
    end
    if isodd(length(θ))
        @inbounds δ = 1 - θ[end]
        -muladd(δ, δ, -s)
    else
        s
    end
end

function QuasiNewtonMethods.∂logdensity!(∇, ::Rosenbrock, θ) 
    s = zero(eltype(θ))
    N = length(θ) >> 1
    @inbounds @simd for i ∈ 1:N
        s -= 100(θ[i+N] - θ[i]^2)^2 + (1 - θ[i])^2
        ∇[i] = 400(θ[i+N] - θ[i]^2)*θ[i] + 2(1 - θ[i])
        ∇[i+N] = 200(θ[i]^2 - θ[i+N])
    end
    if isodd(length(θ))
        @inbounds δ = 1 - θ[end]
        s = -muladd(δ, δ, -s)
        @inbounds ∇[end] = 2δ
    end
    s
end
```
Example usage, and benchmark vs the equivalent method from `Optim.jl`:
```julia
julia> n = 60 # set size
60

julia> state = QuasiNewtonMethods.BFGSState{n}(undef);

julia> x = @StrideArray randn(StaticInt(n));

julia> @test abs(optimize!(state, Rosenbrock(), x)) < eps()
Test Passed

julia> @show QuasiNewtonMethods.optimum(state) .- 1;
QuasiNewtonMethods.optimum(state) .- 1 = [2.751932015598868e-11, 1.3031797863050087e-12, -1.5009105069907491e-12, 2.6655788687435233e-11, -2.244759933489604e-12, -1.579680830587904e-11, -1.1838985347623066e-10, 1.3630208073323047e-11, 1.982880526441022e-11, 5.3439475067307285e-11, -3.896738487441098e-11, 2.4940494114389367e-11, 2.1896706670077037e-11, -2.1127433136314266e-11, 1.4427570249608834e-11, 2.329803017175891e-11, -3.941846848931618e-12, 3.2440716779547074e-13, -5.52979884105298e-12, 1.6714185591126807e-11, -3.831268635678953e-12, 3.045141916402372e-11, 1.3429257705865894e-12, 1.957722872703016e-11, 9.442890913646806e-12, -4.360312111373332e-11, 2.250799546743565e-11, 1.6193268947972683e-11, -1.954936212911207e-11, -7.409961533255682e-12, 5.451172846449026e-11, 2.3572255258841324e-12, -5.270783809407931e-12, 5.5249804731261065e-11, -4.586775403936372e-12, -3.0561220221159147e-11, -2.37073916053987e-10, 2.8603786006442533e-11, 3.746403187676606e-11, 1.092077539510683e-10, -7.855049943827908e-11, 4.777400697264511e-11, 4.427724853428572e-11, -4.329003822078903e-11, 3.108424628806006e-11, 4.3983705566574827e-11, -6.195599588920686e-12, 3.228528555609955e-13, -8.1592510525752e-12, 3.213473931396038e-11, -7.304934435126142e-12, 5.92046411895808e-11, 3.838485085339016e-12, 4.019495847273902e-11, 1.7372103755519674e-11, -8.670275608579914e-11, 4.6741499559743716e-11, 3.335220988276433e-11, -3.7821412668392895e-11, -1.6192935881065296e-11]

julia> @test all(x -> x ≈ 1, QuasiNewtonMethods.optimum(state))
Test Passed

julia> @test maximum(abs, QuasiNewtonMethods.gradient(state)) < 1e-8
Test Passed

julia> @benchmark optimize!($state, Rosenbrock(), $x)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     450.655 μs (0.00% GC)
  median time:      452.243 μs (0.00% GC)
  mean time:        452.309 μs (0.00% GC)
  maximum time:     506.655 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> using Optim, LineSearches

julia> xa = Vector(x);

julia> @benchmark Optim.maximize(x -> QuasiNewtonMethods.logdensity(Rosenbrock(), x), (∇, x) -> QuasiNewtonMethods.∂logdensity!(∇, Rosenbrock(), x), $xa, $(BFGS(linesearch=BackTracking(order=2))))
BenchmarkTools.Trial:
  memory estimate:  291.27 KiB
  allocs estimate:  2483
  --------------
  minimum time:     2.640 ms (0.00% GC)
  median time:      2.657 ms (0.00% GC)
  mean time:        2.673 ms (0.31% GC)
  maximum time:     4.242 ms (34.59% GC)
  --------------
  samples:          1871
  evals/sample:     1
```

Note that in most problems, evaluating the `logdensity` function will be the bottleneck, not the speed of the optimization library itself.
Thus don't expect a performance improvement like this for real problems.
Additionally, QuasiNewtonMethods.jl only provides a backtracking linesearch at the moment. If a different optimization algorithm provides better
results, yielding convergence in fewer function evaluations, then again `Optim.jl` is likely to be faster.


