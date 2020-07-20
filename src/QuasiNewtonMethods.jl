module QuasiNewtonMethods

using SIMDPirates,
    PaddedMatrices,
    Parameters,
    LoopVectorization,
    LinearAlgebra,
    VectorizationBase,
    StackPointers

using PaddedMatrices: AbstractFixedSizeVector,
    AbstractFixedSizeMatrix,
    AbstractFixedSizeArray,
    AbstractFixedSizeVector,
    ConstantVector, ConstantMatrix,
    logdensity, ∂logdensity!

export optimize!, proptimize!


abstract type AbstractProbabilityModel{D} end# <: LogDensityProblems.AbstractLogDensityProblem end
dimension(::AbstractProbabilityModel{D}) where {D} = PaddedMatrices.Static{D}()
Base.length(::AbstractProbabilityModel{D}) where {D} = D
function Base.show(io::IO, ℓ::AbstractProbabilityModel{D}) where {D}
    print(io, "$D-dimensional Probability Model")
end

function update_state!(s, x_old, α)
    @avx for i ∈ eachindex(s)
        sᵢ = α * s[i]
        s[i] = sᵢ
        x_old[i] += sᵢ
    end
end

"""


Updates B⁻¹ and sets yₖ as the new search direction.
"""
function BFGS_update!(B⁻¹::AbstractMatrix{T}, B⁻¹yₖ, yₖ, sₖ, ∇_new, ∇_old) where {T}
    sₖᵀyₖ = zero(T)
    @avx for i ∈ eachindex(sₖ)
        # yₖᵢ = ∇_new[i] - ∇_old[i]
        yₖᵢ = ∇_old[i] - ∇_new[i]
        sₖᵀyₖ += sₖ[i] * yₖᵢ
        yₖ[i] = yₖᵢ
    end
    sₖᵀyₖ⁻¹ = inv(sₖᵀyₖ)
    yₖᵀB⁻¹yₖ = zero(T)
    @avx for c ∈ axes(B⁻¹,2)
        t = zero(yₖᵀB⁻¹yₖ)
        for r ∈ axes(B⁻¹,1)
            t += yₖ[r] * B⁻¹[r,c]
        end
        B⁻¹yₖ[c] = t * sₖᵀyₖ⁻¹
        yₖᵀB⁻¹yₖ += t * yₖ[c]
    end
    # c₁ = (one(T) + yₖᵀB⁻¹yₖ * sₖᵀyₖ⁻¹) * sₖᵀyₖ⁻¹
    c₁ = muladd(yₖᵀB⁻¹yₖ, sₖᵀyₖ⁻¹, one(T)) * sₖᵀyₖ⁻¹
    m = zero(T)
    @avx for c ∈ axes(B⁻¹,2)
        sₙ = zero(eltype(yₖ))
        for r ∈ axes(B⁻¹,1)
            B⁻¹ᵣ = B⁻¹[r,c] + c₁ * sₖ[r] * sₖ[c] - B⁻¹yₖ[r]*sₖ[c] - B⁻¹yₖ[c]*sₖ[r]
            B⁻¹[r,c] = B⁻¹ᵣ
            sₙ += B⁻¹ᵣ * ∇_new[r]
        end
        yₖ[c] = sₙ
        m += sₙ * ∇_new[c]
    end
    m
end


struct BackTracking{O}
    c₁::Float64
    ρₕ::Float64
    ρₗ::Float64
    iterations::Int
end
# BackTracking{O}(c₁ = 1e-4, ρₕ = 0.5, ρₗ = 0.1, iterations = 1_000) where {O} = BackTracking{O}(c₁, ρₕ, ρₗ, iterations)
BackTracking{O}(c₁ = 1e-4, ρₕ = 0.5, ρₗ = 0.1) where {O} = BackTracking{O}(c₁, ρₕ, ρₗ, 1_000)
BackTracking(c₁ = 1e-4, ρₕ = 0.5, ρₗ = 0.1, iterations = 1_000) = BackTracking{3}(c₁, ρₕ, ρₗ, iterations)

abstract type AbstractBFGSState{P,T,L,LT} end

mutable struct BFGSState{P,T,L,LT} <: AbstractBFGSState{P,T,L,LT}
    x_old::ConstantVector{P,T,1,L}
    ∇_new::ConstantVector{P,T,1,L}
    x_new::ConstantVector{P,T,1,L}
    ∇_old::ConstantVector{P,T,1,L}
    y::ConstantVector{P,T,1,L}
    s::ConstantVector{P,T,1,L}
    B⁻¹y::ConstantVector{P,T,1,L}
    B⁻¹::ConstantMatrix{P,P,T,1,L,LT}
    function BFGSState{P,T,L,LT}(::UndefInitializer) where {P,T,L,LT}
        new{P,T,L,LT}()
    end
    @generated function BFGSState{P}(::UndefInitializer) where {P}
        L = PaddedMatrices.calc_padding(P, Float64)
        :(BFGSState{$P,Float64,$L,$(P*L)}(undef))
    end
    @generated function BFGSState{P,T}(::UndefInitializer) where {P,T}
        L = PaddedMatrices.calc_padding(P, T)
        :(BFGSState{$P,$T,$L,$(P*L)}(undef))
    end
end
BFGSState(::Val{P}, ::Type{T} = Float64) where {P,T} = BFGSState{P,T}(undef)
@inline Base.pointer(s::BFGSState{P,T})  where {P,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(s))

#="""
This type exists primarily to be the field of another mutable struct, so that you can get a pointer to this object.
"""
struct ConstantBFGSState{P,T,L,LT} <: AbstractBFGSState{P,T,L,LT}
    x_old::ConstantVector{P,T,L}
    invH::ConstantMatrix{P,P,T,L,LT}
    x_new::ConstantVector{P,T,L}
    ∇_old::ConstantVector{P,T,L}
    δ∇::ConstantVector{P,T,L}
    u::ConstantVector{P,T,L}
    s::ConstantVector{P,T,L}
    ∇::ConstantVector{P,T,L}
end=#
struct PtrBFGSState{P,T,L,LT,NI<:Union{Int,Nothing}} <: AbstractBFGSState{P,T,L,LT}
    ptr::Ptr{T}
    offset::NI
end
@inline Base.pointer(state::PtrBFGSState) = state.ptr

using VectorizationBase: gepbyte
@inline ref_x_old(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(pointer(s))
@inline ref_∇_old(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 4)))
@inline ref_x_new(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 3)))
@inline ref_∇_new(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 2)))
@inline ref_y(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 5)))
@inline ref_s(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 6)))
@inline ref_B⁻¹y(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 7)))
@inline ref_B⁻¹(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrMatrix{P,P}(gepbyte(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 8)))

# @inline ref_x_new(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrVector{P}(gep(pointer(s), s.offset))
# @inline ref_∇_old(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrVector{P}(gep(pointer(s), 2s.offset))
# @inline ref_δ∇(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrVector{P}(gep(pointer(s), 3s.offset))
# @inline ref_u(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrVector{P}(gep(pointer(s), 4s.offset))
# @inline ref_s(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrVector{P}(gep(pointer(s), 5s.offset))
# @inline ref_∇(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrVector{P}(gep(pointer(s), 6s.offset))
# @inline ref_invH(s::PtrBFGSState{-1,T,L,LT,Int}) where {P,T,L,LT} = PtrMatrix{P,P,T,L}(gep(pointer(s), 7s.offset))

function initial_B⁻¹!(B⁻¹::PaddedMatrices.AbstractFixedSizeMatrix{P,P,T,1,L}) where {P,T,L}
    @avx for c ∈ axes(B⁻¹,2), r ∈ axes(B⁻¹,1)
        B⁻¹[r,c] = (r == c)
    end
end
@inline optimum(s::AbstractBFGSState) = ref_x_old(s)

@noinline linesearch_failure(iterations) = error("Linesearch failed to converge, reached maximum iterations $(iterations).")

nanmin(a, b) = a < b ? a : (isnan(b) ? a : b)
nanmax(a, b) = a < b ? b : (isnan(a) ? b : a)
@inline function either_not_finite_else_op(op::F, a, b) where {F}
    isfinite(a) || return true
    isfinite(b) || return true
    op(a, b)
end


fractionalbits(::Type{T}) where {T} = trailing_zeros(reinterpret(Base.uinttype(T),floatmin(T)))
sqrttolerance(::Type{T}) where {T} = T(1 / (1 << (fractionalbits(T) >> 1)))

function step!(x_new, x_old, s, obj, α)
    @avx for i ∈ eachindex(x_new)
        x_new[i] = x_old[i] + α*s[i]
    end
    logdensity(obj, x_new)
end


"""
Optimum value is stored in state.x_old.
"""
function optimize!(state, obj, x::AbstractFixedSizeVector{P,T,L}, ls::BackTracking{order} = BackTracking(), tol = 1e-8) where {P,T,L,order}
    x_old = ref_x_old(state)
    ∇_old = ref_∇_old(state)
    x_new = ref_x_new(state)
    ∇_new = ref_∇_new(state)
    y = ref_y(state)
    s = ref_s(state)
    B⁻¹y = ref_B⁻¹y(state)
    B⁻¹ = ref_B⁻¹(state)
    copyto!(x_old, x)
    # initial_invH!(invH)
    c₁, ρₕ, ρₗ, iterations = T(ls.c₁), T(ls.ρₕ), T(ls.ρₗ), ls.iterations
    iterations = 20
    iterfinitemax = fractionalbits(T)
    sqrttol = sqrttolerance(T)
    α₀ = one(T)
    N = 200
    # f_calls = 0
    # g_calls = 0
    for n ∈ 1:N
        ℓ₀ = ∂logdensity!(∇_new, obj, x_old)#; f_calls +=1; g_calls +=1;
        Base.isfinite(ℓ₀) || return T(NaN)
        # @show (n-1), ℓ₀, maximum(abs, ∇_new), tol ∇_new
        if maximum(abs, ∇_new) < tol
            ∇_new_original = ref_∇_new(state)
            pointer(∇_new) == pointer(∇_new_original) || copyto!(∇_new_original, ∇_new)
            return ℓ₀
        end
        if isone(n)
            m = -one(T) # reset search direction
        else
            # @assert false
            m = BFGS_update!(B⁻¹, B⁻¹y, y, s, ∇_new, ∇_old)
            s, y = y, s
            # @show B⁻¹ x_old
            # @show pointer(s) - pointer(y)
        end
        if m <= zero(T) # If bad, reset search direction
            initial_B⁻¹!(B⁻¹)
            m = zero(T)
            @avx for i ∈ eachindex(x)
                sᵢ = ∇_new[i]
                s[i] = sᵢ
                m += sᵢ * sᵢ
            end
        end
        # @show s
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ℓx₀, ℓx₁ = ℓ₀, ℓ₀
        α₁, α₂ = α₀, α₀
        ℓx₁ = step!(x_new, x_old, s, obj, α₁)

        # Hard-coded backtrack until we find a finite function value
        # Halve α₂ until function value is finite
        iterfinite = 0
        while !Base.isfinite(ℓx₁) && iterfinite < iterfinitemax
            iterfinite += 1
            α₁, α₂ = (α₂, Base.FastMath.mul_fast(T(0.5), α₂))
            ℓx₁ = step!(x_new, x_old, s, obj, α₂)
        end

        # Backtrack until we satisfy sufficient decrease condition
        while !isfinite(ℓx₁) || @fastmath(ℓx₁ < ℓ₀ + α₂*c₁*m)
            # @show α₂, ℓx₁, ℓ₀ + α₂*c₁*m, ℓ₀, α₂, c₁, m
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            iteration > iterations && return T(NaN) # linesearch_failure(iterations)

            # Shrink proposed step-size:
            if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                # https://github.com/JuliaNLSolvers/LineSearches.jl/blob/0c182ce25c32a385a6156304dc24ff72afbfa308/src/backtracking.jl#L84
                αₜ = @fastmath - (m * α₂*α₂) / ( T(2) * (ℓx₁ - ℓ₀ - m*α₂) )
            else
                denom = @fastmath one(T) / (α₁*α₁ * α₂*α₂ * (α₂ - α₁))
                a = @fastmath (α₁*α₁*(ℓx₁ - ℓ₀ - m*α₂) - α₂*α₂*(ℓx₀ - ℓ₀ - m*α₁))*denom
                b = @fastmath (-α₁*α₁*α₁*(ℓx₁ - ℓ₀ - m*α₂) + α₂*α₂*α₂*(ℓx₀ - ℓ₀ - m*α₁))*denom

                if norm(a) <= eps(T) + sqrttol*norm(a)
                    αₜ = @fastmath m / (T(2)*b)
                else
                    # discriminant
                    d = nanmax(@fastmath(b*b - T(3)*a*m), zero(T))
                    # quadratic equation root
                    # αₜ = @fastmath (sqrt(d) - b) / (3a)
                    αₜ = @fastmath (sqrt(d) + b) / (-3a)
                end
            end
            # @show α₂, ℓx₁, ℓ₀ + α₂*c₁*m, ℓ₀, α₂, c₁, m
            α₁ = α₂

            αₜ = nanmin(αₜ, α₂*ρₕ) # avoid too small reductions
            α₂ = nanmax(αₜ, α₂*ρₗ) # avoid too big reductions

            # Evaluate f(x) at proposed position
            ℓx₀, ℓx₁ = ℓx₁, step!(x_new, x_old, s, obj, α₂)
            # @show α₂, ℓx₁, ℓ₀ + α₂*c₁*m, ℓ₀, α₂, c₁, m
        end
        update_state!(s, x_old, α₂)
        ∇_old, ∇_new = ∇_new, ∇_old
        # isone(n) && (s = ref_s(state))
    end
    T(NaN)
end

"""
Optimum value is stored in state.x_old.
"""
function optimize!(_sptr_::StackPointer, obj, x::AbstractFixedSizeVector{P,T,L}, ls::BackTracking{order} = BackTracking(), tol = 1e-8) where {P,T,L,order}
    sptr, x_old = PtrVector{P,T}(_sptr_)
    sptr, ∇ = PtrVector{P,T}(sptr)
    ptr_∇ = pointer(∇)
    _sptr, x_new = PtrVector{P,T}(sptr)
    _sptr, ∇_old = PtrVector{P,T}(_sptr)
    _sptr, invH = PtrMatrix{P,P,T}(_sptr)
    _sptr, δ∇ = PtrVector{P,T}(_sptr)
    _sptr, s = PtrVector{P,T}(_sptr)
    _sptr, u = PtrVector{P,T}(_sptr)
    copyto!(x_old, x)
    initial_invH!(invH)
    c_1, ρ_hi, ρ_lo, iterations = T(ls.c_1), T(ls.ρ_hi), T(ls.ρ_lo), ls.iterations
    iterfinitemax = (round(Int,-log2(eps(T))))
    sqrttol = (Base.FastMath.sqrt_fast(eps(T)))
    α_0 = one(T)
    N = 200
    # f_calls = 0
    # g_calls = 0
    for n ∈ 1:N
        ϕ_0 = - ∂logdensity!(∇, obj, x_old, _sptr)#; f_calls +=1; g_calls +=1;
        Base.isfinite(ϕ_0) || return _sptr_, T(NaN)
        if maximum(abs, ∇) < tol
            if pointer(∇) != ptr_∇
                copyto!(∇_old, ∇)
            end
            return sptr, -ϕ_0
        end
        if n > 1 # update hessian
            dx_dg = zero(T)
            @avx for i ∈ eachindex(∇_old)
                δ∇ₚ =  ∇_old[i] - ∇[i]
                δ∇[i] = δ∇ₚ
                dx_dg += s[i] * δ∇ₚ
            end
            mul!(u, invH, δ∇)
            c2 = one(T) / dx_dg
            c1 = muladd(dot(δ∇, u), c2*c2, c2)
            BFGS_update!(invH, s, u, c1, c2)
        end
        mul!(s, invH, ∇)
        dϕ_0 = zero(T)
        @avx for i ∈ eachindex(s)
            dϕ_0 -= ∇[i] * s[i]
        end
        if dϕ_0 >= zero(T) # If bad, reset search direction
            initial_invH!(invH)
            dϕ_0 = zero(T)
            @avx for i ∈ eachindex(s)
                s_i = ∇[i]
                s[i] = s_i
                dϕ_0 -= s_i * ∇[i]
            end
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        @avx for i ∈ eachindex(s)
            x_new[i] = x_old[i] + α_1*s[i]
        end
        ϕx_1 = -logdensity(obj, x_new, _sptr)#; f_calls += 1;

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !Base.isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = T(0.5)*α_1
            @avx for i ∈ eachindex(s)
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            ϕx_1 = -logdensity(obj, x_new, _sptr)#; f_calls += 1;
        end

        # Backtrack until we satisfy sufficient decrease condition
        @fastmath while either_not_finite_else_op(>, ϕx_1, ϕ_0 + c_1 * α_2 * dϕ_0)
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            iteration > iterations && return _sptr_, T(NaN) # linesearch_failure(iterations)

            # Shrink proposed step-size:
            if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                α_tmp = - (dϕ_0 * α_2*α_2) / ( T(2) * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            else
                div = one(T) / (α_1*α_1 * α_2*α_2 * (α_2 - α_1))
                a = (α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                b = (-α_1*α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2*α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

                if norm(a) <= (eps(T)) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (T(2)*b)
                else
                    # discriminant
                    d = nanmax(b*b - (T(3))*a*dϕ_0, zero(T))
                    # quadratic equation root
                    α_tmp = (sqrt(d) - b) / ((T(3))*a)
                end
            end
            α_1 = α_2

            α_tmp = nanmin(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = nanmax(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            @avx for i ∈ eachindex(s)
                x_new[i] = x_old[i] + α_2*s[i]
            end
            ϕx_0, ϕx_1 = ϕx_1, -logdensity(obj, x_new, _sptr)#; f_calls += 1;
        end
        alpha, fpropose = α_2, ϕx_1
        update_state!(s, x_old, alpha)
        ∇_old, ∇ = ∇, ∇_old
    end
    _sptr_, T(NaN)
end


"""

Optionally pass in the initial density to skip the first ∂logdensity! evaluation

x_old is the initial position.
∇ is a gradient buffer; it will be assumed to hold the initial gradient if the initial value is not nothing.

x_old will be overwritten by the final final position, and ∇ by the final gradient.

"""
function proptimize!(
    sptr::StackPointer, obj::AbstractProbabilityModel{P}, x_old::PtrVector{P,T}, ∇::PtrVector{P,T}, init_nϕ_0::Union{T,Nothing} = nothing,
    penalty::T = 1e-4, N::Int = 200, ls::BackTracking{order} = BackTracking(), tol::T = 1e-8
) where {P,T,order}
    L = VectorizationBase.align(P, T)
    _sptr, x_new = PtrVector{P,T}(sptr)
    _sptr, ∇_old = PtrVector{P,T}(_sptr)
    _sptr, ∇_new = PtrVector{P,T}(_sptr)
    _sptr, invH = PtrMatrix{P,P,T}(_sptr)
    _sptr, δ∇ = PtrVector{P,T}(_sptr)
    _sptr, s = PtrVector{P,T}(_sptr)
    _sptr, u = PtrVector{P,T}(_sptr)
    initial_invH!(invH)
    c_1, ρ_hi, ρ_lo, iterations = T(ls.c_1), T(ls.ρ_hi), T(ls.ρ_lo), ls.iterations
    sqrttol = (Base.FastMath.sqrt_fast(eps(T)))
    α_0 = one(T)
    # @show x_old
    nϕ_0 = init_nϕ_0 ≡ nothing ? ∂logdensity!(∇, obj, x_old, _sptr) : init_nϕ_0
    for n ∈ 1:N
        # @show nϕ_0
        Base.isfinite(nϕ_0) || return T(NaN)
        ϕ_0 = zero(T)
        maxabs∇ = zero(T)
        @avx for i ∈ eachindex(∇)
            xᵢ = x_old[i]
            xᵢpenalty = xᵢ * penalty
            ϕ_0 += xᵢ * xᵢpenalty
            ∇ᵢ = ∇[i] - xᵢpenalty
            ∇_new[i] = ∇ᵢ
            maxabs∇ = max(abs(∇ᵢ), maxabs∇)
        end
        # ϕ_0 = (T(0.5)) * ϕ_0 - nϕ_0
        # @show ϕ_0
        # @show ∇_new
        # @show maxabs∇
        maxabs∇ < tol && return nϕ_0
        ϕ_0 = (T(0.5)) * ϕ_0 - nϕ_0
        if n > 1 # update hessian
            dx_dg = zero(T)
            @avx for i ∈ eachindex(s)
                δ∇_i =  ∇_old[i] - ∇_new[i]
                δ∇[i] = δ∇_i
                dx_dg += s[i] * δ∇_i
            end
            mul!(u, invH, δ∇)
            c2 = one(T) / dx_dg
            c1 = muladd(dot(δ∇, u), c2*c2, c2)
            BFGS_update!(invH, s, u, c1, c2)
        end
        mul!(s, invH, ∇_new)
        dϕ_0 = zero(T)
        @avx for i ∈ eachindex(s)
            s_i = s[i]
            dϕ_0 -= ∇_new[i] * s_i
        end
        if dϕ_0 >= zero(T) # If bad, reset search direction
            initial_invH!(invH)
            dϕ_0 = zero(T)
            @avx for i ∈ eachindex(s)
                s_i = ∇_new[i]
                s[i] = s_i
                dϕ_0 -= s_i * ∇_new[i]
            end
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        @avx for i ∈ eachindex(s)
            x_new[i] = x_old[i] + α_1*s[i]
        end
        # SIMDArrays.vadd!(x_new, x_old, α_1, s)
        # ϕx_1 = f(x + α_1*s); f_calls += 1;
        nϕx_1 = logdensity(obj, x_new, _sptr)#; f_calls += 1;
        # @show nϕx_1
        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !Base.isfinite(nϕx_1) && iterfinite < (round(Int,-log2(eps(T))))
            iterfinite += 1
            α_1 = α_2
            α_2 = T(0.5)*α_1
            @avx for i ∈ eachindex(s)
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # @show iterfinite
            # @show x_new
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            # ϕx_1 = f(x + α_2*s); f_calls += 1;
            nϕx_1 = logdensity(obj, x_new, _sptr)#; f_calls += 1;
        end
        ϕx_1 = zero(T)
        @avx for i ∈ eachindex(x_new)
            xᵢ = x_new[i]
            ϕx_1 += xᵢ * xᵢ
        end
        @fastmath ϕx_1 = (T(0.5))*penalty*ϕx_1 - nϕx_1
        # Backtrack until we satisfy sufficient decrease condition
        @fastmath while either_not_finite_else_op(>, ϕx_1, ϕ_0 + c_1 * α_2 * dϕ_0)
            # Increment the number of steps we've had to perform
            iteration += 1
            # @show iteration, iterations
            # Ensure termination
            iteration > iterations && return T(NaN) # linesearch_failure(iterations)

            # Shrink proposed step-size:
            if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                α_tmp = - (dϕ_0 * α_2*α_2) / ( T(2) * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            else
                div = one(T) / (α_1*α_1 * α_2*α_2 * (α_2 - α_1))
                a = (α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                b = (-α_1*α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2*α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

                if norm(a) <= (eps(T)) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (T(2)*b)
                else
                    # discriminant
                    d = nanmax(b*b - (T(3))*a*dϕ_0, zero(T))
                    # quadratic equation root
                    α_tmp = (sqrt(d) - b) / ((T(3))*a)
                end
            end
            α_1 = α_2

            α_tmp = nanmin(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = nanmax(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            # ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
            @avx for i ∈ eachindex(s)
                x_new[i] = x_old[i] + α_2*s[i]
            end
            ϕx_0, nϕx_1 = ϕx_1, logdensity(obj, x_new, _sptr)#; f_calls += 1;
            ϕx_1 = zero(T)
            @avx for i ∈ eachindex(x_new)
                xᵢ = x_new[i]
                ϕx_1 += xᵢ * xᵢ
            end
            @fastmath ϕx_1 = (T(0.5))*penalty*ϕx_1 - nϕx_1
        end
        alpha, fpropose = α_2, ϕx_1
        update_state!(s, x_old, alpha)
        ∇_old, ∇_new = ∇_new, ∇_old
        nϕ_0 = ∂logdensity!(∇, obj, x_old, _sptr)#; f_calls +=1; g_calls +=1;
        # println("at end, nϕ_0: nϕ_0")
        # @show x_old
        # @show x_new
        # @show reinterpret(Int,pointer(x_new)) - reinterpret(Int,pointer(x_old))
    end
    # println("Reached maximum number of iterations: N.")
    # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), x_old
    # T(NaN)
    nϕ_0
end


# include("precompile.jl")
# _precompile_()


end # module
