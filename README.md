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
  Expression: abs(optimize!(state, Rosenbrock(), x)) < eps()
   Evaluated: 1.1276127615598287e-18 < 2.220446049250313e-16

julia> @show QuasiNewtonMethods.optimum(state) .- 1;
QuasiNewtonMethods.optimum(state) .- 1 = [-4.3435033347805074e-11, -6.733913426870686e-11, 2.8338220658952196e-11, -2.003880394951807e-10, -8.865719269834926e-11, 2.7613467068476893e-11, -1.0049250320776082e-10, 5.030487137958062e-11, 7.431388837630948e-12, -1.941459215615282e-10, 2.8667734852660942e-11, 4.852851454018037e-11, -2.883271399412024e-11, -1.0002554340360348e-11, 3.650657554032932e-11, 6.867306723279398e-11, -2.1006529848932587e-11, 3.950129112695322e-11, -8.582179411575908e-11, 3.096656264744979e-11, -3.50218742894981e-11, -6.686190490157173e-10, 1.8915313759748642e-11, -1.389866000067741e-11, 3.703792827991492e-11, -3.067912590637434e-11, 7.048515104912667e-10, 4.9040327354532565e-11, -9.172385073696887e-11, -3.8369973864860185e-11, -8.825984387783592e-11, -1.3295142764491175e-10, 6.662714824301474e-11, -3.998605890842555e-10, -1.7559209641859752e-10, 5.617106779709502e-11, -2.0395707345244318e-10, 1.037505636958258e-10, 7.899236820207989e-12, -3.8792036249901685e-10, 5.822275994660231e-11, 1.0021161678253065e-10, -6.63278321155758e-11, -2.3326562903491777e-11, 7.75075559289462e-11, 1.3814993593541658e-10, -4.3182679654307776e-11, 7.76778641409237e-11, -1.6486345622013232e-10, 6.214295744655374e-11, -6.747724601297023e-11, -1.3378324004165165e-9, 3.717626206878322e-11, -2.401823184783325e-11, 7.553246916813805e-11, -5.6644355872492724e-11, 1.4112184754111468e-9, 9.812284318400089e-11, -1.8162560344592293e-10, -8.00064459127725e-11]

julia> @test QuasiNewtonMethods.optimum(state) ≈ fill(1, n)
Test Passed
  Expression: QuasiNewtonMethods.optimum(state) ≈ fill(1, n)
   Evaluated: [0.999999999956565, 0.9999999999326609, 1.0000000000283382, 0.999999999799612, 0.9999999999113428, 1.0000000000276135, 0.9999999998995075, 1.0000000000503049, 1.0000000000074314, 0.9999999998058541  …  0.9999999999325228, 0.9999999986621676, 1.0000000000371763, 0.9999999999759818, 1.0000000000755325, 0.9999999999433556, 1.0000000014112185, 1.0000000000981228, 0.9999999998183744, 0.9999999999199936] ≈ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1  …  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

julia> @test maximum(abs, QuasiNewtonMethods.gradient(state)) < 1e-8
Test Passed
  Expression: maximum(abs, QuasiNewtonMethods.gradient(state)) < 1.0e-8
   Evaluated: 3.923606328839031e-9 < 1.0e-8

julia> @benchmark optimize!($state, Rosenbrock(), $x)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  329.524 μs … 376.496 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     330.825 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   331.036 μs ±   1.215 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                ▁▂▄▅▇▇▇▇██▆▄▄▃▂           ▁▁▁▂
  ▂▁▁▂▂▂▃▃▃▃▄▅▆▇█████████████████▆▆▆▆▇▇▆█▇██████▇█▇▆▅▅▄▄▄▃▃▃▃▃▂ ▅
  330 μs           Histogram: frequency by time          333 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> using Optim, LineSearches

julia> xa = Vector(x);

julia> @benchmark Optim.maximize(x -> QuasiNewtonMethods.logdensity(Rosenbrock(), x), (∇, x) -> QuasiNewtonMethods.∂logdensity!(∇, Rosenbrock(), x), $xa, $(BFGS(linesearch=BackTracking(order=2))))
BenchmarkTools.Trial: 1461 samples with 1 evaluation.
 Range (min … max):  3.329 ms …   5.712 ms  ┊ GC (min … max): 0.00% … 36.28%
 Time  (median):     3.375 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.419 ms ± 241.028 μs  ┊ GC (mean ± σ):  0.93% ±  4.45%

  ▅█▆▂     ▁
  ████▅▁▁▆▇██▆▁▁▁▁▃▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▅▅▅ █
  3.33 ms      Histogram: log(frequency) by time      5.05 ms <

 Memory estimate: 893.39 KiB, allocs estimate: 4182.
```

Note that in most problems, evaluating the `logdensity` function will be the bottleneck, not the speed of the optimization library itself.
Thus don't expect a performance improvement like this for real problems.
Additionally, QuasiNewtonMethods.jl only provides a backtracking linesearch at the moment. If a different optimization algorithm provides better
results, yielding convergence in fewer function evaluations, then again `Optim.jl` is likely to be faster.


