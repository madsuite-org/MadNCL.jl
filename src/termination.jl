#=
    Termination on MadNLP's criterion for the original problem
=#

"""
    MadNLPTermination(nlp; tol, s_max, max_gradient)

MadNLP's stopping test applied to the original problem `nlp` at an iterate of the NCL loop,
so that `madncl` with `termination = :madnlp` returns points that satisfy exactly what
`madnlp(nlp; tol)` requires: the scaled primal infeasibility, the scaled dual infeasibility
and the complementarity, each at most `tol` (`MadNLP.get_inf_pr`, `get_inf_du`,
`get_inf_compl` with `get_sd` / `get_sc`), on the problem scaled as MadNLP scales it (objective
and constraint factors from the gradients at `x0`, at most 1, `max_gradient` over the
largest entry).

The original problem is read in MadNLP's form: equality rows `c_i(x) = l_i`, inequality rows
with a slack `s_i = c_i(x)` bounded in `[l_i, u_i]`, bound multipliers of the slack derived
from the row multiplier (`y = zu_s - zl_s`), and every bound relaxed by
`bound_relax_factor * max(1, |bound|)` as MadNLP relaxes them (the slack bounds in the scaled
units it works in); the scaling factors are taken at the initial point pushed inside the
relaxed bounds (`bound_push`, `bound_fac`), where MadNLP takes them. Fixed variables are left out, as MadNLP removes them. Every operation is
a broadcast or a reduction, so the test runs on the GPU.
"""
struct MadNLPTermination{T, VT}
    tol::T
    s_max::T
    obj_scale::T
    con_scale::VT
    lvar::VT
    uvar::VT
    lcon::VT
    ucon::VT
    free_var::VT      # 1 for a variable with lvar < uvar, 0 for a fixed one
    ineq::VT          # 1 for an inequality row (a slack in MadNLP's form), 0 for an equality row
    # work buffers
    x::VT; y::VT; zl::VT; zu::VT          # the iterate in the original problem (n, m, n, n)
    g::VT; c::VT; jtv::VT                 # ∇f(x), c(x), Jᵀy
    cs::VT                                # c(x) in MadNLP's scaled units
end

function MadNLPTermination(nlp::NLPModels.AbstractNLPModel{T, VT}; tol, s_max = T(100), max_gradient = T(100),
                           bound_relax = T(1e-8), bound_push = T(1e-2), bound_fac = T(1e-2)) where {T, VT}
    n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
    lvar0, uvar0 = NLPModels.get_lvar(nlp), NLPModels.get_uvar(nlp)
    relax_lo(b) = b .- max.(one(T), abs.(b)) .* bound_relax
    relax_up(b) = b .+ max.(one(T), abs.(b)) .* bound_relax
    lvar, uvar = relax_lo(lvar0), relax_up(uvar0)
    # the scaling factors are taken where MadNLP takes them: at the initial point pushed
    # inside the relaxed bounds
    x0 = copy(NLPModels.get_x0(nlp))
    MadNLP.initialize_variables!(x0, lvar, uvar, bound_push, bound_fac)
    g = NLPModels.grad(nlp, x0)
    obj_scale = min(one(T), max_gradient / max(norm(g, Inf), eps(T)))
    con_scale = fill!(similar(x0, m), one(T))
    if m > 0   # the Jacobian in coordinate form, as MadNLP's sparse callback reads it
        nnzj = NLPModels.get_nnzj(nlp)
        Ji = similar(x0, Int, nnzj); Jj = similar(x0, Int, nnzj)
        NLPModels.jac_structure!(nlp, Ji, Jj)
        jac = similar(x0, nnzj)
        NLPModels.jac_coord!(nlp, x0, jac)
        MadNLP.set_con_scale_sparse!(con_scale, Ji, jac, max_gradient)
    end
    free_var = T.(lvar0 .< uvar0)
    # the slack bounds are relaxed in MadNLP's scaled units and kept there
    lcon = relax_lo(con_scale .* NLPModels.get_lcon(nlp)) ./ con_scale
    ucon = relax_up(con_scale .* NLPModels.get_ucon(nlp)) ./ con_scale
    mk(k) = fill!(similar(x0, k), zero(T))
    return MadNLPTermination{T, VT}(
        T(tol), s_max, obj_scale, con_scale, lvar, uvar, lcon, ucon, free_var,
        T.(NLPModels.get_lcon(nlp) .< NLPModels.get_ucon(nlp)),
        mk(n), mk(m), mk(n), mk(n), mk(n), mk(m), mk(n), mk(m),
    )
end

# the current iterate of the NCL solver in the original problem: x, and multipliers with
# ∇f + Jᵀy - zl + zu = 0 for the objective MadNLP minimizes (f, or -f for a maximization)
function original_iterate!(term::MadNLPTermination{T, VT}, solver::NCLSolver{T, VT}) where {T, VT}
    ipm, ncl = solver.ipm, solver.ncl
    # the unpacked vectors hold every variable of the model, fixed ones included, then the
    # m relaxation variables; solver.n counts only the variables MadNLP kept
    n, m = NLPModels.get_nvar(ncl.nlp), solver.m
    xr = similar(VT, n + m); zl = similar(VT, n + m); zu = similar(VT, n + m)
    MadNLP.unpack_x!(xr, ipm.cb, MadNLP.variable(ipm.x))
    MadNLP.unpack_z!(zl, ipm.cb, MadNLP.variable(ipm.zl))
    MadNLP.unpack_z!(zu, ipm.cb, MadNLP.variable(ipm.zu))
    y = copy(ipm.y)
    MadNLP.update_z!(ipm.cb, xr, y, zl, zu, ipm.jacl)
    if isa(ncl.nlp, ScaledModel)
        y .= y .* ncl.nlp.scaling_cons ./ ncl.nlp.scaling_obj
        zl .= zl ./ ncl.nlp.scaling_obj
        zu .= zu ./ ncl.nlp.scaling_obj
    end
    term.x .= @view xr[1:n]
    term.y .= y
    term.zl .= @view zl[1:n]
    term.zu .= @view zu[1:n]
    return term
end

original_model(ncl::NCLModel) = ncl.nlp isa ScaledModel ? ncl.nlp.nlp : ncl.nlp

"""
    madnlp_residuals(term, nlp, x, y, zl, zu) -> (inf_pr, inf_du, inf_compl)

MadNLP's three termination residuals of `nlp` at `(x, y, zl, zu)`, the multipliers in the
convention of `NLPModels` (`∇f + Jᵀy - zl + zu = 0` for the objective MadNLP minimizes).
"""
function madnlp_residuals(term::MadNLPTermination{T}, nlp, x, y, zl, zu) where {T}
    σf, σc, s_max = term.obj_scale, term.con_scale, term.s_max
    sgn = NLPModels.get_minimize(nlp) ? one(T) : -one(T)
    m = length(y)
    # primal: every row in MadNLP's scaled units; an equality row against its right-hand side,
    # an inequality row against its (relaxed) slack bounds
    if m > 0
        NLPModels.cons!(nlp, x, term.c)
        term.cs .= σc .* term.c
        rhs = σc .* NLPModels.get_lcon(nlp)
        viol_s = ifelse.(term.ineq .> 0, max.(σc .* term.lcon .- term.cs, term.cs .- σc .* term.ucon, zero(T)),
                         abs.(term.cs .- rhs))
        inf_pr = maximum(viol_s)
    else
        inf_pr = zero(T)
    end
    # dual: MadNLP's scaled multipliers are y·σf/σc on the rows and σf·z on the bounds
    NLPModels.grad!(nlp, x, term.g)
    if m > 0
        NLPModels.jtprod!(nlp, x, y, term.jtv)
    else
        fill!(term.jtv, zero(T))
    end
    res = σf .* (sgn .* term.g .+ term.jtv .- zl .+ zu) .* term.free_var
    ym = y .* σf ./ σc
    zls = max.(.-ym, zero(T)); zus = max.(ym, zero(T))                 # slack bound multipliers
    lb_x = T.(term.lvar .> -Inf) .* term.free_var; ub_x = T.(term.uvar .< Inf) .* term.free_var
    lb_s = T.(term.lcon .> -Inf) .* term.ineq                           # inequality rows only
    ub_s = T.(term.ucon .< Inf) .* term.ineq
    zlm = σf .* zl .* lb_x; zum = σf .* zu .* ub_x
    zls = zls .* lb_s; zus = zus .* ub_s
    n_l = sum(lb_x) + sum(lb_s); n_u = sum(ub_x) + sum(ub_s)
    sd = max(s_max, (norm(ym, 1) + norm(zlm, 1) + norm(zum, 1) + norm(zls, 1) + norm(zus, 1)) / max(one(T), m + n_l + n_u)) / s_max
    sc = max(s_max, (norm(zlm, 1) + norm(zum, 1) + norm(zls, 1) + norm(zus, 1)) / max(one(T), n_l + n_u)) / s_max
    inf_du = maximum(abs, res; init = zero(T)) / sd
    # complementarity with μ = 0; a slack sits at c(x) clamped into its bounds. An infinite
    # bound has no multiplier: its term is set to zero rather than evaluated (Inf * 0 is NaN).
    compl_x = max.(ifelse.(lb_x .> 0, abs.((x .- term.lvar) .* zlm), zero(T)),
                   ifelse.(ub_x .> 0, abs.((term.uvar .- x) .* zum), zero(T)))
    inf_compl = maximum(compl_x; init = zero(T))
    if m > 0
        s = clamp.(term.cs, σc .* term.lcon, σc .* term.ucon)
        compl_s = max.(ifelse.(lb_s .> 0, abs.((s .- σc .* term.lcon) .* zls), zero(T)),
                       ifelse.(ub_s .> 0, abs.((σc .* term.ucon .- s) .* zus), zero(T)))
        inf_compl = max(inf_compl, maximum(compl_s; init = zero(T)))
    end
    return inf_pr, inf_du, inf_compl / sc
end

function madnlp_converged!(term::MadNLPTermination{T}, solver::NCLSolver{T}) where {T}
    original_iterate!(term, solver)
    inf_pr, inf_du, inf_compl = madnlp_residuals(term, original_model(solver.ncl), term.x, term.y, term.zl, term.zu)
    return max(inf_pr, inf_du, inf_compl) <= term.tol
end
