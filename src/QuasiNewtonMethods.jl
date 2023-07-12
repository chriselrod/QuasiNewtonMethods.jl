module QuasiNewtonMethods

using StrideArraysCore, LoopVectorization, VectorizationBase

using VectorizationBase: gep
using StrideArraysCore: PtrArray, StrideArray, AbstractStrideArray

function logdensity end
function ∂logdensity! end

export optimize!


abstract type AbstractProbabilityModel{D} end# <: LogDensityProblems.AbstractLogDensityProblem end
dimension(::AbstractProbabilityModel{D}) where {D} = StaticInt{D}()
Base.length(::AbstractProbabilityModel{D}) where {D} = D
function Base.show(io::IO, ℓ::AbstractProbabilityModel{D}) where {D}
  print(io, "$D-dimensional Probability Model")
end

function update_state!(s, x_old, α)
  @turbo for i ∈ eachindex(s)
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
  @turbo for i ∈ eachindex(sₖ)
    # yₖᵢ = ∇_new[i] - ∇_old[i]
    yₖᵢ = ∇_old[i] - ∇_new[i]
    sₖᵀyₖ += sₖ[i] * yₖᵢ
    yₖ[i] = yₖᵢ
  end
  sₖᵀyₖ⁻¹ = inv(sₖᵀyₖ)
  yₖᵀB⁻¹yₖ = zero(T)
  @turbo for c ∈ axes(B⁻¹, 2)
    t = zero(yₖᵀB⁻¹yₖ)
    for r ∈ axes(B⁻¹, 1)
      # t += yₖ[r] * B⁻¹[r,c]
      t += yₖ[r] * B⁻¹[c, r]
    end
    B⁻¹yₖ[c] = t * sₖᵀyₖ⁻¹
    yₖᵀB⁻¹yₖ += t * yₖ[c]
  end
  # c₁ = (one(T) + yₖᵀB⁻¹yₖ * sₖᵀyₖ⁻¹) * sₖᵀyₖ⁻¹
  c₁ = muladd(yₖᵀB⁻¹yₖ, sₖᵀyₖ⁻¹, one(T)) * sₖᵀyₖ⁻¹
  m = zero(T)
  @turbo for c ∈ axes(B⁻¹, 2)
    sₙ = zero(eltype(yₖ))
    for r ∈ axes(B⁻¹, 1)
      # B⁻¹ᵣ = B⁻¹[r,c] + c₁ * sₖ[r] * sₖ[c] - B⁻¹yₖ[r]*sₖ[c] - B⁻¹yₖ[c]*sₖ[r]
      # B⁻¹[r,c] = B⁻¹ᵣ
      B⁻¹ᵣ = B⁻¹[c, r] + c₁ * sₖ[r] * sₖ[c] - B⁻¹yₖ[r] * sₖ[c] - B⁻¹yₖ[c] * sₖ[r]
      B⁻¹[c, r] = B⁻¹ᵣ
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
BackTracking{O}(c₁=1e-4, ρₕ=0.5, ρₗ=0.1) where {O} = BackTracking{O}(c₁, ρₕ, ρₗ, 1_000)
BackTracking(c₁=1e-4, ρₕ=0.5, ρₗ=0.1, iterations=1_000) = BackTracking{2}(c₁, ρₕ, ρₗ, iterations)

abstract type AbstractBFGSState{P,T,L,LT} end

mutable struct BFGSState{P,T,L,LT} <: AbstractBFGSState{P,T,L,LT}
  x_old::NTuple{L,T}
  ∇_new::NTuple{L,T}
  x_new::NTuple{L,T}
  ∇_old::NTuple{L,T}
  y::NTuple{L,T}
  s::NTuple{L,T}
  B⁻¹y::NTuple{L,T}
  B⁻¹::NTuple{LT,T}
  function BFGSState{P,T,L,LT}(::UndefInitializer) where {P,T,L,LT}
    new{P,T,L,LT}()
  end
  @generated function BFGSState{P}(::UndefInitializer) where {P}
    L = VectorizationBase.align(P, VectorizationBase.pick_vector_width(Val(P), Float64))
    :(BFGSState{$P,Float64,$L,$(P * L)}(undef))
  end
  @generated function BFGSState{P,T}(::UndefInitializer) where {P,T}
    L = VectorizationBase.align(P, VectorizationBase.pick_vector_width(Val(P), T))
    :(BFGSState{$P,$T,$L,$(P * L)}(undef))
  end
end
BFGSState(::Val{P}, ::Type{T}=Float64) where {P,T} = BFGSState{P,T}(undef)
@inline Base.pointer(s::BFGSState{P,T}) where {P,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(s))

function VectorizationBase.lifetime_start!(state::AbstractBFGSState{P,T,L}) where {P,T,L}
  nothing
  # VectorizationBase.lifetime_start!(pointer(ref_x_new(state)), Val(VectorizationBase.staticmul(T, StaticInt{L}()*(StaticInt{5}()+StaticInt{L}()))))
end
function VectorizationBase.lifetime_end!(state::AbstractBFGSState{P,T,L}) where {P,T,L}
  nothing
  # VectorizationBase.lifetime_end!(pointer(ref_x_new(state)), Val(VectorizationBase.staticmul(T, StaticInt{L}()*(StaticInt{5}()+StaticInt{L}()))))
end

struct PtrBFGSState{P,T,L,LT,NI<:Union{Int,Nothing}} <: AbstractBFGSState{P,T,L,LT}
  ptr::Ptr{T}
  offset::NI
end
@inline Base.pointer(state::PtrBFGSState) = state.ptr

@inline function dense_vector(ptr::Ptr{T}, ::StaticInt{P}) where {T,P}
  PtrArray(ptr, (StaticInt{P}(),), (static(1),), Val{(true,)}())
end
@inline function square_matrix(ptr::Ptr{T}, ::StaticInt{P}, ::StaticInt{X}) where {T,P,X}
  st = static(1)
  PtrArray(ptr, (StaticInt{P}(), StaticInt{P}()), (st, st * StaticInt{X}()), Val{(true, false)}())
end
@inline function square_matrix(ptr::Ptr{T}, ::StaticInt{P}, ::StaticInt{P}) where {T,P}
  st = static(1)
  PtrArray(ptr, (StaticInt{P}(), StaticInt{P}()), (st, st * StaticInt{P}()), Val{(true, true)}())
end

@inline ref_x_old(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(pointer(s), StaticInt{P}())
@inline ref_∇_old(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 4)), StaticInt{P}())
@inline ref_x_new(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 3)), StaticInt{P}())
@inline ref_∇_new(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 2)), StaticInt{P}())
@inline ref_y(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 5)), StaticInt{P}())
@inline ref_s(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 6)), StaticInt{P}())
@inline ref_B⁻¹y(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = dense_vector(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 7)), StaticInt{P}())
@inline ref_B⁻¹(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = square_matrix(gep(pointer(s), fieldoffset(BFGSState{P,T,L,LT}, 8)), StaticInt{P}(), StaticInt{L}())

function initial_B⁻¹!(B⁻¹::AbstractMatrix)
  @turbo for c ∈ axes(B⁻¹, 2), r ∈ axes(B⁻¹, 1)
    B⁻¹[r, c] = (r == c)
  end
end
@inline optimum(s::AbstractBFGSState) = ref_x_old(s)
@inline gradient(s::AbstractBFGSState) = ref_∇_new(s)

nanmin(a, b) = a < b ? a : (isnan(b) ? a : b)
nanmax(a, b) = a < b ? b : (isnan(a) ? b : a)

sqrttolerance(::Type{T}) where {T} = T(1 / (1 << (Base.Math.significand_bits(T) >> 1)))

function step!(x_new, x_old, s, obj, α)
  @turbo for i ∈ eachindex(x_new)
    x_new[i] = x_old[i] + α * s[i]
  end
  logdensity(obj, x_new)
end


function linesearch!(x_new::AbstractVector{T}, x_old, s, obj, ℓ₀, m, ls::BackTracking{order}) where {T,order}
  c₁, ρₕ, ρₗ, iterations = T(ls.c₁), T(ls.ρₕ), T(ls.ρₗ), ls.iterations
  sqrttol = sqrttolerance(T)

  α₀ = one(T)

  # Count the total number of iterations
  ℓx₀, ℓx₁ = ℓ₀, ℓ₀
  α₁, α₂ = α₀, α₀
  ℓx₁ = step!(x_new, x_old, s, obj, α₁)

  # Hard-coded backtrack until we find a finite function value
  # Halve α₂ until function value is finite
  iterfinite = 0
  iterfinitemax = Base.Math.significand_bits(T)
  while !isfinite(ℓx₁) && iterfinite < iterfinitemax
    iterfinite += 1
    α₁, α₂ = (α₂, Base.FastMath.mul_fast(T(0.5), α₂))
    ℓx₁ = step!(x_new, x_old, s, obj, α₂)
  end
  iteration = 0
  # Backtrack until we satisfy sufficient decrease condition
  while !(ℓx₁ ≥ @fastmath(ℓ₀ + α₂ * c₁ * m))
    # @show α₂, ℓx₁, ℓ₀ + α₂*c₁*m, ℓ₀, α₂, c₁, m
    # Increment the number of steps we've had to perform
    iteration += 1

    # Ensure termination
    iteration > iterations && return zero(T) # linesearch_failure(iterations)

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
      αₜ = @fastmath -(m * α₂ * α₂) / (T(2) * (ℓx₁ - ℓ₀ - m * α₂))
    else
      denom = @fastmath one(T) / (α₁ * α₁ * α₂ * α₂ * (α₂ - α₁))
      a = @fastmath (α₁ * α₁ * (ℓx₁ - ℓ₀ - m * α₂) - α₂ * α₂ * (ℓx₀ - ℓ₀ - m * α₁)) * denom
      b = @fastmath (-α₁ * α₁ * α₁ * (ℓx₁ - ℓ₀ - m * α₂) + α₂ * α₂ * α₂ * (ℓx₀ - ℓ₀ - m * α₁)) * denom

      if abs(a) <= eps(T) + sqrttol * abs(a)
        αₜ = @fastmath m / (T(2) * b)
      else
        # discriminant
        d = nanmax(@fastmath(b * b - T(3) * a * m), zero(T))
        # quadratic equation root
        # αₜ = @fastmath (sqrt(d) - b) / (3a)
        αₜ = @fastmath (sqrt(d) + b) / (-3a)
      end
    end
    # @show α₂, ℓx₁, ℓ₀ + α₂*c₁*m, ℓ₀, α₂, c₁, m
    α₁ = α₂

    αₜ = nanmin(αₜ, α₂ * ρₕ) # avoid too small reductions
    α₂ = nanmax(αₜ, α₂ * ρₗ) # avoid too big reductions

    # Evaluate f(x) at proposed position
    ℓx₀, ℓx₁ = ℓx₁, step!(x_new, x_old, s, obj, α₂)
    # @show α₂, ℓx₁, ℓ₀ + α₂*c₁*m, ℓ₀, α₂, c₁, m
  end
  α₂
end

"""
Optimum value is stored in state.x_old.
"""
function optimize!(state, obj, x::AbstractStrideArray{T}, ls=BackTracking(), tol=1e-8) where {T}
  GC.@preserve state obj x begin
    VectorizationBase.lifetime_start!(state)
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
    N = 10_000
    # f_calls = 0
    # g_calls = 0
    for n ∈ 1:N
      ℓ₀ = ∂logdensity!(∇_new, obj, x_old)#; f_calls +=1; g_calls +=1;
      Base.isfinite(ℓ₀) || break
      # @show (n-1), ℓ₀, maximum(abs, ∇_new), tol ∇_new
      if maximum(abs, ∇_new) < tol
        ∇_new_original = ref_∇_new(state)
        pointer(∇_new) == pointer(∇_new_original) || copyto!(∇_new_original, ∇_new)
        VectorizationBase.lifetime_end!(state)
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
        @turbo for i ∈ eachindex(x)
          sᵢ = ∇_new[i]
          s[i] = sᵢ
          m += sᵢ * sᵢ
        end
      end
      # @show s
      #### Perform line search
      α₂ = linesearch!(x_new, x_old, s, obj, ℓ₀, m, ls)
      iszero(α₂) && break
      update_state!(s, x_old, α₂)
      ∇_old, ∇_new = ∇_new, ∇_old
      # isone(n) && (s = ref_s(state))
    end
    VectorizationBase.lifetime_end!(state)
  end
  T(NaN)
end

# include("precompile.jl")
# _precompile_()


end # module
