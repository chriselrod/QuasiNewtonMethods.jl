using QuasiNewtonMethods, StrideArrays, Aqua
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

@testset "QuasiNewtonMethods.jl" begin
    Aqua.test_all(QuasiNewtonMethods)
    for n ∈ 2:24
        @show n
        state = QuasiNewtonMethods.BFGSState{n}(undef);
        x = @StrideArray randn(StaticInt(n));
        # 2nd order
        @test abs(optimize!(state, Rosenbrock(), x)) < 2eps()
        @show QuasiNewtonMethods.optimum(state) .- 1;
        @test all(x -> x ≈ 1, QuasiNewtonMethods.optimum(state))
        @test maximum(abs, QuasiNewtonMethods.gradient(state)) < 1e-8
        # 3rd order
        @test abs(optimize!(state, Rosenbrock(), x, QuasiNewtonMethods.BackTracking{3}())) < eps()
        @show QuasiNewtonMethods.optimum(state) .- 1;
        @test all(x -> x ≈ 1, QuasiNewtonMethods.optimum(state))
        @test maximum(abs, QuasiNewtonMethods.gradient(state)) < 1e-8
    end
end


