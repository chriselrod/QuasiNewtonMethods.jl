using QuasiNewtonMethods, PaddedMatrices
using Test

struct Rosenbrock end
function QuasiNewtonMethods.logdensity(::Rosenbrock, θ)
    s = zero(eltype(θ))
    @inbounds @simd for i ∈ 1:length(θ)-1
        s -= 100(θ[i+1] - θ[i]^2)^2 + (1 - θ[i])^2
    end
    s
end

function QuasiNewtonMethods.∂logdensity!(∇, ::Rosenbrock, θ) 
    s = zero(eltype(θ))
    ∇[firstindex(∇)] = 0
    @inbounds @simd for i ∈ 1:length(∇)-1
        s -= 100(θ[i+1] - θ[i]^2)^2 + (1 - θ[i])^2
        ∇[i] += 400(θ[i+1] - θ[i]^2)*θ[i] + 2(1 - θ[i])
        ∇[i+1] = 200(θ[i]^2 - θ[i+1])
    end
    s
end

@testset "QuasiNewtonMethods.jl" begin
    # Write your own tests here.
    state = QuasiNewtonMethods.BFGSState{2}(undef);
    x = @FixedSize rand(2);
    @test abs(optimize!(state, Rosenbrock(), x)) < eps()
    @test all(isapprox(1), QuasiNewtonMethods.optimum(state))
    @test maximum(abs, QuasiNewtonMethods.gradient(state)) < 1e-8
end


