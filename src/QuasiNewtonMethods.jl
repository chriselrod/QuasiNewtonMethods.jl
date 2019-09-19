module QuasiNewtonMethods

using   SIMDPirates,
    PaddedMatrices,
    Parameters,
    LinearAlgebra,
    VectorizationBase,
    StackPointers

using PaddedMatrices: AbstractMutableFixedSizePaddedVector,
    AbstractMutableFixedSizePaddedMatrix,
    AbstractFixedSizePaddedArray,
    AbstractFixedSizePaddedVector

export optimize!

"""
To find a mode, define methods for `logdensity` and logdensity_and_gradient!` dispatching on obj, and evaluating at the position `q`.

logdensity(obj, q, [::StackPointer])
logdensity_and_gradient!(∇, obj, q, [::StackPointer])

These must return a value (eg, a logdensity). logdensity_and_gradient! should store the gradient in ∇.
"""
function logdensity end
function logdensity_and_gradient! end
#=
function BFGS_update_quote(R,stride,T)
    W, rows, cols, row_reps_per_kernel = PaddedMatrices.pick_kernel_size(T, R, R)
    V = Vec{W,T}
    common = quote
        vC1 = vbroadcast($V, c1)
        vC2 = vbroadcast($V, -c2)
        ptr_invH, ptr_S, ptr_U = pointer(invH), pointer(s), pointer(u)
    end
    rowreps, rowrem = divrem(stride, rows)
    colreps, colrem = divrem(R, rows)
    
    
end
function BFGS_update_block_quote(W, T, row_iter, col_iter, stride)
    V = Vec{W,T}
    size_T = sizeof(T)
    WT = size_T*W
    q = quote end
    for c ∈ 0:col_iter-1
    end
    for r ∈ 0:row_iter-1
        row_setup = quote
            $(Symbol(:vU_,r)) = vload($V, ptr_U + $WT*$r)
            $(Symbol(:vS_,r)) = vload($V, ptr_S + $WT*$r)
        end
        push!(q.args, row_setup)
    end
    for c ∈ 0:col_iter-1
        vSB_c = $(Symbol(:vSb_,c))
        vSBc1 = $(Symbol(:vSbc1_,c))
        vSBc2 = $(Symbol(:vSbc2_,c))
        vUbc2 = $(Symbol(:vUBc_,c))
        col_setup = quote
            $vSB_c = vbroadcast($V, VectorizationBase.load(ptr_S + $c*$size_T ))
            $vSBc1 = vmul($vSb_c, vC1)
            $vSBc2 = vmul($vSb_c, vC2)
            $vUBc2 = vmul(vbroadcast($V, VectorizationBase.load(ptr_U + $c*$size_T )), vC2)
        end
        push!(q.args, col_setup)
        for r ∈ 0:row_iter-1
            push!(q.args, Expr(:(=), Symbol(:invH_,r,:_,c), :(vload($V, ptr_invH + $WT*$r + $stride*$c))))
        end
        for r ∈ 0:row_iter-1
            invHrc = Symbol(:invH_,r,:_,c)
            push!(q.args, :($invHrc = vmuladd($(Symbol(:vU_,r)),$vSBc2,$invHrc)))
        end
        for r ∈ 0:row_iter-1
            invHrc = Symbol(:invH_,r,:_,c)
            push!(q.args, :($invHrc = vmuladd($(Symbol(:vS_,r)),$vUBc2,$invHrc)))
        end
        for r ∈ 0:row_iter-1
            invHrc = Symbol(:invH_,r,:_,c)
            push!(q.args, :($invHrc = vmuladd($(Symbol(:vS_,r)),$vSBc1,$invHrc)))
        end
        for r ∈ 0:row_iter-1
            push!(q.args, :(vstore!(ptr_invH + $WT*$r + $stride*$c, $(Symbol(:invH_,r,:_,c)))))
        end
    end
end
=#
function bfgs_column_update_block(W, T, row_iter, stride, c, ptr_S = :ptr_S, ptr_U = :ptr_U)
    V = Vec{W,T}
    WT = sizeof(T)*W
    q = quote end
    for r ∈ 0:row_iter-1
        push!(q.args, :($(Symbol(:invH_,r)) = vload($V, ptr_invH + $WT*$r + $stride*$c)))
        push!(q.args, :($(Symbol(:vU_,r)) = vload($V, $ptr_U + $WT*$r)))
    end
    for r ∈ 0:row_iter-1
        invHrc = Symbol(:invH_,r)
        push!(q.args, :($(Symbol(:vS_,r)) = vload($V, $ptr_S + $WT*$r)))
        push!(q.args, :($invHrc = vmuladd($(Symbol(:vU_,r)),vSbc2,$invHrc)))
    end
    for r ∈ 0:row_iter-1
        invHrc = Symbol(:invH_,r)
        push!(q.args, :($invHrc = vmuladd($(Symbol(:vS_,r)),vUbc2,$invHrc)))
    end
    for r ∈ 0:row_iter-1
        invHrc = Symbol(:invH_,r)
        push!(q.args, :($invHrc = vmuladd($(Symbol(:vS_,r)),vSbc1,$invHrc)))
    end
    for r ∈ 0:row_iter-1
        push!(q.args, :(vstore!(ptr_invH + $WT*$r + $stride*$c, $(Symbol(:invH_,r)))))
    end
    q
end


function BFGS_update_quote(R,stride,T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(R, T)
    size_T = sizeof(T)
    stride_bytes = stride * size_T
    WT = W * size_T
    Q = stride >> Wshift
    if Q > 0
        stride & (W-1) == 0 || throw("Number of rows plus padding $stride_AD not a multiple of register size: $(VectorizationBase.REGISTER_SIZE).")
    end
    V = Vec{W,T}
    common = quote
        vC1 = vbroadcast($V, c1)
        vC2 = vbroadcast($V, -c2)
        ptr_invH = pointer(invH); ptr_S = pointer(s); ptr_U = pointer(u)
    end
    qreps, qrem = divrem(Q, 4)
    if Q % (qreps+1) == 0
        qreps, qrem = qreps+1, 0
        rr = Q ÷ qreps
    else
        rr = 4
    end
    if qreps > 1
        row_block = quote
            ptrS_rb = ptr_S; ptrU_rb = ptr_U
            for rb ∈ 0:$(qreps-1)
                $(bfgs_column_update_block(W, T, rr, stride_bytes, :c, :ptrS_rb, :ptrU_rb))
                ptrS_rb += $WT*$rr; ptrU_rb += $WT*$rr
            end
        end
        push!(row_block.args, bfgs_column_update_block(W, T, qrem, stride_bytes, :c, :ptrS_rb, :ptrU_rb))
    elseif qreps == 1
        row_block = quote
            ptrS_rb = ptr_S; ptrU_rb = ptr_U
            $(bfgs_column_update_block(W, T, rr, stride_bytes, :c, :ptrS_rb, :ptrU_rb))
            ptrS_rb += $WT*$rr; ptrU_rb += $WT*$rr
            $(bfgs_column_update_block(W, T, qrem, stride_bytes, :c, :ptrS_rb, :ptrU_rb))
        end
    else
        row_block = bfgs_column_update_block(W, T, qrem, stride_bytes, :c)
    end
    quote
        $common
        for c ∈ 0:$(R-1)
            vSb = vbroadcast($V, VectorizationBase.load(ptr_S + c*$size_T ))
            vSbc1 = vmul(vSb, vC1)
            vSbc2 = vmul(vSb, vC2)
            vUbc2 = vmul(vbroadcast($V, VectorizationBase.load(ptr_U + c*$size_T )), vC2)
            $row_block
        end
        nothing
    end
end

@generated function BFGS_update!(invH::Union{Symmetric{T,<:AbstractMutableFixedSizePaddedMatrix{P,P,T,R,L}},<:AbstractMutableFixedSizePaddedMatrix{P,P,T,R,L}},
    s::AbstractMutableFixedSizePaddedVector{P,T,R}, u::AbstractMutableFixedSizePaddedVector{P,T,R}, c1::T, c2::T) where {P,T,L,R}

    BFGS_update_quote(P,R,T)

end



struct BackTracking{O}
    c_1::Float64
    ρ_hi::Float64
    ρ_lo::Float64
    iterations::Int
end
function BackTracking(::Val{O} = Val(3); c_1 = 1e-4, ρ_hi = 0.5, ρ_lo = 0.1, iterations = 1_000) where {O}
    BackTracking{O}(c_1, ρ_hi, ρ_lo, iterations, maxstep)
end

abstract type AbstractBFGSState{P,T,L,LT} end

mutable struct BFGSState{P,T,L,LT} <: AbstractBFGSState{P,T,L,LT}
    x_old::ConstantFixedSizePaddedVector{P,T,L,L}
    invH::ConstantFixedSizePaddedMatrix{P,P,T,L,LT}
    x_new::ConstantFixedSizePaddedVector{P,T,L,L}
    ∇_old::ConstantFixedSizePaddedVector{P,T,L,L}
    δ∇::ConstantFixedSizePaddedVector{P,T,L,L}
    u::ConstantFixedSizePaddedVector{P,T,L,L}
    s::ConstantFixedSizePaddedVector{P,T,L,L}
    ∇::ConstantFixedSizePaddedVector{P,T,L,L}
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
    invH::ConstantFixedSizePaddedMatrix{P,P,T,L,LT}
    x_old::ConstantFixedSizePaddedVector{P,T,L,L}
    x_new::ConstantFixedSizePaddedVector{P,T,L,L}
    ∇_old::ConstantFixedSizePaddedVector{P,T,L,L}
    # ∇_new::SizedSIMDVector{P,T,L}
    δ∇::ConstantFixedSizePaddedVector{P,T,L,L}
    u::ConstantFixedSizePaddedVector{P,T,L,L}
    s::ConstantFixedSizePaddedVector{P,T,L,L}
end=#
struct PtrBFGSState{P,T,L,LT} <: AbstractBFGSState{P,T,L,LT}
    ptr::Ptr{T}
end
@inline Base.pointer(state::PtrBFGSState) = state.ptr


@inline ref_x_old(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s))# + LT*sizeof(T))
@inline ref_invH(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrMatrix{P,P,T,L,LT,false}(pointer(s) + L*sizeof(T))
@inline ref_x_new(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + (LT+L)*sizeof(T))
@inline ref_∇_old(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + (LT+2L)*sizeof(T))
@inline ref_δ∇(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + (LT+3L)*sizeof(T))
@inline ref_u(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + (LT+4L)*sizeof(T))
@inline ref_s(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + (LT+5L)*sizeof(T))
@inline ref_∇(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + (LT+6L)*sizeof(T))

function initial_invH!(invH::PaddedMatrices.AbstractFixedSizePaddedMatrix{P,P,T}) where {P,T}
    fill!(invH, zero(T))
    @inbounds for p = 1:P
        invH[p,p] = one(T)
    end
end
@inline optimum(s::AbstractBFGSState{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L,false}(pointer(s) + LT*sizeof(T))

@noinline linesearch_failure(iterations) = error("Linesearch failed to converge, reached maximum iterations $(iterations).")

nanmin(a, b) = a < b ? a : (isnan(b) ? a : b)
nanmax(a, b) = a < b ? b : (isnan(a) ? b : a)


@generated function update_state!(C::AbstractFixedSizePaddedArray{S,T,N,R,L}, B::AbstractFixedSizePaddedArray{S,T,N,R,L}, α::T) where {S,T,N,R,L}
    T_size = sizeof(T)
    VL = min(VectorizationBase.REGISTER_SIZE ÷ T_size, L)
    VLT = VL * T_size
    V = Vec{VL,T}
    iter = L ÷ VL
    q = quote
        ptr_C = pointer(C)
        ptr_B = pointer(B)
        vα = vbroadcast($V, α)
    end
    rep, rem = divrem(iter, 4)
    if (rep == 1 && rem == 0) || (rep >= 1 && rem != 0)
        rep -= 1
        rem += 4
    end
    if rep > 0
        push!(q.args,
            quote
              for iunscaled ∈ 0:$(rep-1)
                    i = $(4VLT)*iunscaled
                    vs_0 = vmul(vload($V, ptr_C + i), vα)
                    vstore!(ptr_C + i, vs_0)
                    vstore!(ptr_B + i, vadd(vload($V, ptr_B + i), vs_0))

                    vs_1 = vmul(vload($V, ptr_C + i + $VLT), vα)
                    vstore!(ptr_C + i + $VLT, vs_1)
                    vstore!(ptr_B + i + $VLT, vadd(vload($V, ptr_B + i + $VLT), vs_1))

                    vs_2 = vmul(vload($V, ptr_C + i + $(2VLT)), vα)
                    vstore!(ptr_C + i + $(2VLT), vs_2)
                    vstore!(ptr_B + i + $(2VLT), vadd(vload($V, ptr_B + i + $(2VLT)), vs_2))

                    vs_3 = vmul(vload($V, ptr_C + i + $(3VLT)), vα)
                    vstore!(ptr_C + i + $(3VLT), vs_3)
                    vstore!(ptr_B + i + $(3VLT), vadd(vload($V, ptr_B + i + $(3VLT)), vs_3))
                end
            end
        )
    end
    for i ∈ 0:(rem-1)
        offset = VLT*(i + 4rep)
        push!(q.args,
            quote
                $(Symbol(:vs_,i)) = vmul(vload($V, ptr_C + $offset), vα)
                vstore!(ptr_C + $offset, $(Symbol(:vs_,i)))
                vstore!(ptr_B + $offset, vadd(vload($V, ptr_B + $offset), $(Symbol(:vs_,i))))
            end
        )
    end

    push!(q.args, :(nothing))
    q
end


"""
Optimum value is stored in state.x_old.
"""
function optimize!(state, obj, x::AbstractFixedSizePaddedVector{P,T,L}, ls::BackTracking{order} = BackTracking(), tol = 1e-8) where {P,T,L,order}
    x_old = ref_x_old(state)
    ∇_old = ref_∇_old(state)
    invH = ref_invH(state)
    δ∇ = ref_δ∇(state)
    s = ref_s(state)
    u = ref_u(state)
    ∇ = ref_∇(state)
    x_new = ref_x_new(state)
    copyto!(x_old, x)
    initial_invH!(invH)
    c_1, ρ_hi, ρ_lo, iterations = T(ls.c_1), T(ls.ρ_hi), T(ls.ρ_lo), ls.iterations
    iterfinitemax = round(Int,-log2(eps(T)))
    sqrttol = sqrt(eps(T))
    α_0 = one(T)
    N = 200
    f_calls = 0
    g_calls = 0
    @fastmath for n ∈ 1:N
        ϕ_0 = - logdensity_and_gradient!(∇, obj, x_old); f_calls +=1; g_calls +=1;
        isfinite(ϕ_0) || return T(NaN)
        if maximum(abs, ∇) < tol
            if pointer(∇) != pointer(ref_∇(state))
                copyto!(∇_old, ∇)
            end
            return -ϕ_0
        end
        if n > 1 # update hessian
            dx_dg = zero(T)
            @inbounds @simd for i ∈ 1:L
                δ∇_i =  ∇_old[i] - ∇[i]
                δ∇[i] = δ∇_i
                dx_dg += s[i] * δ∇_i
            end
            mul!(u, invH, δ∇)
            c2 = one(T) / dx_dg
            c1 = muladd(dot(δ∇, u), c2*c2, c2)
            BFGS_update!(invH, s, u, c1, c2)
        end
        mul!(s, invH, ∇)
        dϕ_0 = zero(T)
        @inbounds @simd for i ∈ 1:L
            s_i = s[i]
            dϕ_0 -= ∇[i] * s_i
        end
        if dϕ_0 >= zero(T) # If bad, reset search direction
            initial_invH!(invH)
            dϕ_0 = zero(T)
            @inbounds @simd for i ∈ 1:L
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
        @inbounds @simd for i ∈ 1:L
            x_new[i] = x_old[i] + α_1*s[i]
        end
        # SIMDArrays.vadd!(x_new, x_old, α_1, s)
        # ϕx_1 = f(x + α_1*s); f_calls += 1;
        ϕx_1 = -logdensity(obj, x_new); f_calls += 1;

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = T(0.5)*α_1
            @inbounds @simd for i ∈ 1:L
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            # ϕx_1 = f(x + α_2*s); f_calls += 1;
            ϕx_1 = -logdensity(obj, x_new); f_calls += 1;
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
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

                if norm(a) <= eps(T) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (T(2)*b)
                else
                    # discriminant
                    d = max(b*b - T(3)*a*dϕ_0, zero(T))
                    # quadratic equation root
                    α_tmp = (sqrt(d) - b) / (T(3)*a)
                end
            end
            α_1 = α_2

            α_tmp = nanmin(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = nanmax(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            # ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
            @inbounds @simd for i ∈ 1:L
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            ϕx_0, ϕx_1 = ϕx_1, -logdensity(obj, x_new); f_calls += 1;
        end
        alpha, fpropose = α_2, ϕx_1
        update_state!(s, x_old, alpha)
        ∇_old, ∇ = ∇, ∇_old
#        copyto!(∇_old, ∇)

    end
    # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), x_old
    T(NaN)
end

"""
Optimum value is stored in state.x_old.
"""
@generated function optimize!(_sptr_::StackPointer, obj, x::AbstractFixedSizePaddedVector{P,T,L}, ls::BackTracking{order} = BackTracking(), tol = 1e-8) where {P,T,L,order}
    quote
        sptr, x_old = PtrVector{$P,$T}(_sptr_)
        sptr, ∇ = PtrVector{$P,$T}(sptr)
        ptr_∇ = pointer(∇)
        _sptr, x_new = PtrVector{$P,$T}(sptr)
        _sptr, ∇_old = PtrVector{$P,$T}(_sptr)
        _sptr, invH = PtrMatrix{$P,$P,$T}(_sptr)
        _sptr, δ∇ = PtrVector{$P,$T}(_sptr)
        _sptr, s = PtrVector{$P,$T}(_sptr)
        _sptr, u = PtrVector{$P,$T}(_sptr)
        copyto!(x_old, x)
        initial_invH!(invH)
        c_1, ρ_hi, ρ_lo, iterations = T(ls.c_1), T(ls.ρ_hi), T(ls.ρ_lo), ls.iterations
        iterfinitemax = $(round(Int,-log2(eps(T))))
        sqrttol = $(Base.FastMath.sqrt_fast(eps(T)))
        α_0 = one($T)
        N = 200
        f_calls = 0
        g_calls = 0
        @fastmath for n ∈ 1:N
            ϕ_0 = - logdensity_and_gradient!(∇, obj, x_old, _sptr); f_calls +=1; g_calls +=1;
            isfinite(ϕ_0) || return _sptr_, $T(NaN)
            if maximum(abs, ∇) < tol
                if pointer(∇) != ptr_∇
                    copyto!(∇_old, ∇)
                end
                return sptr, -ϕ_0
            end
            if n > 1 # update hessian
                dx_dg = zero($T)
                $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                    δ∇_i =  ∇_old[i] - ∇[i]
                    δ∇[i] = δ∇_i
                    dx_dg += s[i] * δ∇_i
                end end))
                mul!(u, invH, δ∇)
                c2 = one($T) / dx_dg
                c1 = muladd(dot(δ∇, u), c2*c2, c2)
                BFGS_update!(invH, s, u, c1, c2)
            end
            mul!(s, invH, ∇)
            dϕ_0 = zero($T)
            @inbounds @simd for i ∈ 1:$L
                s_i = s[i]
                dϕ_0 -= ∇[i] * s_i
            end
            if dϕ_0 >= zero($T) # If bad, reset search direction
                initial_invH!(invH)
                dϕ_0 = zero(T)
                $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                    s_i = ∇[i]
                    s[i] = s_i
                    dϕ_0 -= s_i * ∇[i]
                end end))
            end
            #### Perform line search

            # Count the total number of iterations
            iteration = 0
            ϕx_0, ϕx_1 = ϕ_0, ϕ_0
            α_1, α_2 = α_0, α_0
            @inbounds @simd for i ∈ 1:$L
                x_new[i] = x_old[i] + α_1*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_1, s)
            # ϕx_1 = f(x + α_1*s); f_calls += 1;
            ϕx_1 = -logdensity(obj, x_new, _sptr); f_calls += 1;

            # Hard-coded backtrack until we find a finite function value
            iterfinite = 0
            while !isfinite(ϕx_1) && iterfinite < iterfinitemax
                iterfinite += 1
                α_1 = α_2
                α_2 = T(0.5)*α_1
                @inbounds @simd for i ∈ 1:$L
                    x_new[i] = x_old[i] + α_2*s[i]
                end
                # SIMDArrays.vadd!(x_new, x_old, α_2, s)
                # ϕx_1 = f(x + α_2*s); f_calls += 1;
                ϕx_1 = -logdensity(obj, x_new, _sptr); f_calls += 1;
            end

            # Backtrack until we satisfy sufficient decrease condition
            while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
                # Increment the number of steps we've had to perform
                iteration += 1

                # Ensure termination
                iteration > iterations && return _sptr_, $T(NaN) # linesearch_failure(iterations)

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

                    if norm(a) <= $(eps(T)) + sqrttol*norm(a)
                        α_tmp = dϕ_0 / (T(2)*b)
                    else
                        # discriminant
                        d = max(b*b - $(T(3))*a*dϕ_0, zero($T))
                        # quadratic equation root
                        α_tmp = (sqrt(d) - b) / ($(T(3))*a)
                    end
                end
                α_1 = α_2

                α_tmp = nanmin(α_tmp, α_2*ρ_hi) # avoid too small reductions
                α_2 = nanmax(α_tmp, α_2*ρ_lo) # avoid too big reductions

                # Evaluate f(x) at proposed position
                # ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
                @inbounds @simd for i ∈ 1:$L
                    x_new[i] = x_old[i] + α_2*s[i]
                end
                ϕx_0, ϕx_1 = ϕx_1, -logdensity(obj, x_new, _sptr); f_calls += 1;
            end
            alpha, fpropose = α_2, ϕx_1
            update_state!(s, x_old, alpha)
            ∇_old, ∇ = ∇, ∇_old
        end
        # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), x_old
        _sptr_, $T(NaN)
    end
end



@generated function proptimize!(_sptr_::StackPointer, obj, x::AbstractFixedSizePaddedVector{P,T,L}, penalty::T = 1e-4, N::Int = 200, ls::BackTracking{order} = BackTracking(), tol = 1e-8) where {P,T,L,order}
    W = VectorizationBase.pick_vector_width(P, T)
    quote
        sptr, x_old = PtrVector{$P,$T}(_sptr_)
        sptr, ∇ = PtrVector{$P,$T}(sptr)
        ptr_∇ = pointer(∇)
        _sptr, x_new = PtrVector{$P,$T}(sptr)
        _sptr, ∇_old = PtrVector{$P,$T}(_sptr)
        _sptr, ∇_new = PtrVector{$P,$T}(_sptr)
        _sptr, invH = PtrMatrix{$P,$P,$T}(_sptr)
        _sptr, δ∇ = PtrVector{$P,$T}(_sptr)
        _sptr, s = PtrVector{$P,$T}(_sptr)
        _sptr, u = PtrVector{$P,$T}(_sptr)
        copyto!(x_old, x)
        initial_invH!(invH)
        c_1, ρ_hi, ρ_lo, iterations = T(ls.c_1), T(ls.ρ_hi), T(ls.ρ_lo), ls.iterations
        iterfinitemax = $(round(Int,-log2(eps(T))))
        sqrttol = $(Base.FastMath.sqrt_fast(eps(T)))
        α_0 = one($T)
        # N = 200
        f_calls = 0
        g_calls = 0
        @fastmath for n ∈ 1:N
            nϕ_0 = logdensity_and_gradient!(∇, obj, x_old, _sptr); f_calls +=1; g_calls +=1;
            isfinite(ϕ_0) || return _sptr_, $T(NaN)
            ϕ_0 = zero($T)
            vmaxabs∇ = vbroadcast(Vec{$W,$T}, zero($T))
            $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                xᵢ = x_old[i]
                xᵢpenalty = xᵢ * penalty
                ϕ_0 += xᵢ * xᵢpenalty
                ∇ᵢ = ∇[i] - xᵢpenalty
                ∇_new[i] = ∇ᵢ
                vmaxabs∇ = SIMDPirates.vmax(SIMDPirates.vabs(∇ᵢ),vmaxabs∇)
            end end))
            vany(SIMDPirates.vgreater(vmaxabs∇, tol)) && return sptr, nϕ_0
            ϕ_0 = $(T(0.5)) * ϕ_0 - nϕ_0
            if n > 1 # update hessian
                dx_dg = zero($T)
                $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                    δ∇_i =  ∇_old[i] - ∇[i]
                    δ∇[i] = δ∇_i
                    dx_dg += s[i] * δ∇_i
                end end))
                mul!(u, invH, δ∇)
                c2 = one($T) / dx_dg
                c1 = muladd(dot(δ∇, u), c2*c2, c2)
                BFGS_update!(invH, s, u, c1, c2)
            end
            mul!(s, invH, ∇)
            dϕ_0 = zero($T)
            @inbounds @simd for i ∈ 1:$L
                s_i = s[i]
                dϕ_0 -= ∇[i] * s_i
            end
            if dϕ_0 >= zero($T) # If bad, reset search direction
                initial_invH!(invH)
                dϕ_0 = zero(T)
                $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                    s_i = ∇[i]
                    s[i] = s_i
                    dϕ_0 -= s_i * ∇[i]
                end end))
            end
            #### Perform line search

            # Count the total number of iterations
            iteration = 0
            ϕx_0, ϕx_1 = ϕ_0, ϕ_0
            α_1, α_2 = α_0, α_0
            @inbounds @simd for i ∈ 1:$L
                x_new[i] = x_old[i] + α_1*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_1, s)
            # ϕx_1 = f(x + α_1*s); f_calls += 1;
            nϕx_1 = logdensity(obj, x_new, _sptr); f_calls += 1;

            # Hard-coded backtrack until we find a finite function value
            iterfinite = 0
            while !isfinite(nϕx_1) && iterfinite < iterfinitemax
                iterfinite += 1
                α_1 = α_2
                α_2 = T(0.5)*α_1
                @inbounds @simd for i ∈ 1:$L
                    x_new[i] = x_old[i] + α_2*s[i]
                end
                # SIMDArrays.vadd!(x_new, x_old, α_2, s)
                # ϕx_1 = f(x + α_2*s); f_calls += 1;
                nϕx_1 = logdensity(obj, x_new, _sptr); f_calls += 1;
            end
            ϕx_1 = zero(T)
            $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                          xᵢ = x_new[i]
                          ϕx_1 += xᵢ * xᵢ
                          end end))
            ϕx_1 = $(T(0.5))*penalty*ϕx_1 - nϕx_1
            # Backtrack until we satisfy sufficient decrease condition
            while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
                # Increment the number of steps we've had to perform
                iteration += 1

                # Ensure termination
                iteration > iterations && return _sptr_, $T(NaN) # linesearch_failure(iterations)

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

                    if norm(a) <= $(eps(T)) + sqrttol*norm(a)
                        α_tmp = dϕ_0 / (T(2)*b)
                    else
                        # discriminant
                        d = max(b*b - $(T(3))*a*dϕ_0, zero($T))
                        # quadratic equation root
                        α_tmp = (sqrt(d) - b) / ($(T(3))*a)
                    end
                end
                α_1 = α_2

                α_tmp = nanmin(α_tmp, α_2*ρ_hi) # avoid too small reductions
                α_2 = nanmax(α_tmp, α_2*ρ_lo) # avoid too big reductions

                # Evaluate f(x) at proposed position
                # ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
                @inbounds @simd for i ∈ 1:$L
                    x_new[i] = x_old[i] + α_2*s[i]
                end
                ϕx_0, nϕx_1 = ϕx_1, logdensity(obj, x_new, _sptr); f_calls += 1;
                ϕx_1 = zero(T)
                $(macroexpand(LoopVectorization, quote @vvectorize $T for i ∈ 1:$P
                              xᵢ = x_new[i]
                              ϕx_1 += xᵢ * xᵢ
                              end end))
                ϕx_1 = $(T(0.5))*penalty*ϕx_1 - nϕx_1
            end
            alpha, fpropose = α_2, ϕx_1
            update_state!(s, x_old, alpha)
            ∇_old, ∇ = ∇, ∇_old
        end
        # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), x_old
        _sptr_, $T(NaN)
    end
end




end # module
