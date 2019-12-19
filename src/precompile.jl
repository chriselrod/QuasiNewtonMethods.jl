function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(QuasiNewtonMethods.BFGS_update_quote),Int64,Int64,Type{T} where T})
    precompile(Tuple{typeof(QuasiNewtonMethods.bfgs_column_update_block),Int64,Type{T} where T,Int64,Int64,Symbol,Symbol,Symbol})
end
