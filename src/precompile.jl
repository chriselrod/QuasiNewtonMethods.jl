function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    # precompile(Tuple{QuasiNewtonMethods.var"##s39#7",Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any})
    precompile(Tuple{typeof(QuasiNewtonMethods.BFGS_update_quote),Int64,Int64,Type})
end
