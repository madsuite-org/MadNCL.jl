

"""
    ScaledModel

Scale the nonlinear program
```
min_x        f(x)
subject to   c♭ ≤ c(x) ≤ c♯
             x ≥ 0

```
as
```
min_x        σf f(x)
subject to   σc . c♭ ≤ σc . c(x) ≤ σc . c♯
             x ≥ 0

```
with ``σf`` a scalar defined as
```
σf = min(1, max_gradient / norm(g0, Inf))

```
and ``σc`` a vector whose size is equal to the number of constraints in the model.
For ``i=1, ..., m``,
```
σc[i] = min(1, max_gradient / norm(J0[i, :], Inf))

```

The vector ``g0 = ∇f(x0)`` and the matrix ``J0 = ∇c(x0)`` are resp.
the gradient and the Jacobian evaluated at the initial point ``x0``.

"""
struct ScaledModel{T, VT, M} <: NLPModels.AbstractNLPModel{T, VT}
    nlp::M
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    scaling_obj::T
    scaling_cons::VT # [size m]
    scaling_jac::VT  # [size nnzj]
    buffer_cons::VT  # [size m]
end

function ScaledModel(
    nlp::NLPModels.AbstractNLPModel{T, VT};
    max_gradient=T(1),
) where {T, VT}
    n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
    nnzj = NLPModels.get_nnzj(nlp)
    x0 = NLPModels.get_x0(nlp)

    buffer_cons  = VT(undef, m)
    scaling_cons = VT(undef, m)
    scaling_jac  = VT(undef, nnzj)
    fill!(scaling_cons, zero(T))

    # Scale objective
    g = NLPModels.grad(nlp, x0)
    scaling_obj = min(one(T), max_gradient / norm(g, Inf))

    # Scale constraints
    Ji = similar(x0, Int, nnzj)
    Jj = similar(x0, Int, nnzj)
    NLPModels.jac_structure!(nlp, Ji, Jj)
    NLPModels.jac_coord!(nlp, x0, scaling_jac)
    MadNLP.set_con_scale_sparse!(scaling_cons, Ji, scaling_jac, max_gradient)
    MadNLP.set_jac_scale_sparse!(scaling_jac, scaling_cons, Ji)

    meta = NLPModels.NLPModelMeta(
        n;
        lvar=NLPModels.get_lvar(nlp),
        uvar=NLPModels.get_uvar(nlp),
        x0=x0,
        y0=NLPModels.get_y0(nlp) .* scaling_obj ./ scaling_cons,
        nnzj=NLPModels.get_nnzj(nlp),
        nnzh=NLPModels.get_nnzh(nlp),
        ncon=m,
        lcon=NLPModels.get_lcon(nlp) .* scaling_cons,
        ucon=NLPModels.get_ucon(nlp) .* scaling_cons,
        minimize=NLPModels.get_minimize(nlp),
    )

    return ScaledModel(
        nlp,
        meta,
        NLPModels.Counters(),
        scaling_obj,
        scaling_cons,
        scaling_jac,
        buffer_cons,
    )
end

function NLPModels.obj(scaled::ScaledModel{T, VT}, x::AbstractVector) where {T, VT <: AbstractVector{T}}
    return scaled.scaling_obj * NLPModels.obj(scaled.nlp, x)
end

function NLPModels.cons!(scaled::ScaledModel, x::AbstractVector, c::AbstractVector)
    NLPModels.cons!(scaled.nlp, x, c)
    c .*= scaled.scaling_cons
    return c
end

function NLPModels.grad!(scaled::ScaledModel, x::AbstractVector, g::AbstractVector)
    NLPModels.grad!(scaled.nlp, x, g)
    g .*= scaled.scaling_obj
    return g
end

function NLPModels.jprod!(scaled::ScaledModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    NLPModels.jprod!(scaled.nlp, x, v, Jv)
    Jv .*= scaled.scaling_cons
    return Jv
end

function NLPModels.jtprod!(scaled::ScaledModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    v_scaled = scaled.buffer_cons
    v_scaled .= v .* scaled.scaling_cons
    NLPModels.jtprod!(scaled.nlp, x, v_scaled, Jtv)
    return Jtv
end

function NLPModels.jac_structure!(scaled::ScaledModel, jrows::AbstractVector, jcols::AbstractVector)
    return NLPModels.jac_structure!(scaled.nlp, jrows, jcols)
end

function NLPModels.jac_coord!(scaled::ScaledModel, x::AbstractVector, jac::AbstractVector)
    NLPModels.jac_coord!(scaled.nlp, x, jac)
    jac .*= scaled.scaling_jac
    return jac
end

function NLPModels.hess_structure!(scaled::ScaledModel, hrows::AbstractVector, hcols::AbstractVector)
    return NLPModels.hess_structure!(scaled.nlp, hrows, hcols)
end

function NLPModels.hess_coord!(
    scaled::ScaledModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    hess::AbstractVector{T};
    obj_weight::T=one(T),
) where T
    y_scaled = scaled.buffer_cons
    y_scaled .= y .* scaled.scaling_cons
    σ = obj_weight * scaled.scaling_obj
    NLPModels.hess_coord!(scaled.nlp, x, y_scaled, hess; obj_weight=σ)
    return hess
end
