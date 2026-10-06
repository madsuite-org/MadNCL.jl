#=
    NCLModel.

    Adapted from the NCLModel in: https://github.com/JuliaSmoothOptimizers/NCL.jl/blob/main/src/NCLModel.jl

=#

@doc raw"""
    NCLModel

Reformulate the nonlinear program
```
min_x        f(x)
subject to   c♭ ≤ c(x) ≤ c♯
             x ≥ 0

```
using an augmented-Lagrangian formulation. For a given scalar ``ρ``
and multiplier ``y_e``, the NCL model writes
```
min_{x,r}    f(x) - y_eᵀ r + \frac{ρ}{2} \| r \|^2
subject to   c♭ ≤ c(x) + r ≤ c♯
             x ≥ 0

```
"""
struct NCLModel{T, VT, M} <: NLPModels.AbstractNLPModel{T, VT}
    nlp::M
    nx::Int
    nr::Int
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    yk::VT
    ρk::Base.RefValue{T}
end

function NCLModel(
    nlp::NLPModels.AbstractNLPModel{T, VT};
    resid::T=zero(T),
    ρ::T=one(T),
) where {T, VT}

    nx = NLPModels.get_nvar(nlp)
    nr = NLPModels.get_ncon(nlp)
    nvar = nx + nr

    y = fill!(similar(VT, nr), one(T))

    lvarx = NLPModels.get_lvar(nlp)
    uvarx = NLPModels.get_uvar(nlp)

    lvarr = fill!(similar(VT, nr), -Inf)
    uvarr = fill!(similar(VT, nr),  Inf)

    x0 = NLPModels.get_x0(nlp)
    r0 = fill!(similar(VT, nr), resid)

    meta = NLPModels.NLPModelMeta{T, VT}(
        nvar;
        lvar=vcat(lvarx, lvarr),
        uvar=vcat(uvarx, uvarr),
        x0=vcat(x0, r0),
        y0=NLPModels.get_y0(nlp),
        nnzj=NLPModels.get_nnzj(nlp) + nr,
        nnzh=NLPModels.get_nnzh(nlp) + nr,
        ncon=nr,
        lcon=NLPModels.get_lcon(nlp),
        ucon=NLPModels.get_ucon(nlp),
        minimize=true,
    )
    cnt = NLPModels.Counters()

    return NCLModel{T, VT, typeof(nlp)}(
        nlp, nx, nr, meta, cnt, y, Ref{T}(ρ),
    )
end

set_barrier!(ncl::NCLModel, μ) = nothing

function NLPModels.obj(ncl::NCLModel{T, VT}, xr::VT) where {T, VT <: AbstractVector{T}}
    sense = NLPModels.get_minimize(ncl.nlp) ? T(1) : T(-1)
    x = view(xr, 1:ncl.nx)
    r = view(xr, 1+ncl.nx:ncl.nx+ncl.nr)
    obj_val = NLPModels.obj(ncl.nlp, x)
    obj_res = -dot(ncl.yk, r) + ncl.ρk[] * dot(r, r) / T(2)
    return sense * obj_val + obj_res
end

function NLPModels.cons!(ncl::NCLModel{T, VT}, xr::VT, cx::VT) where {T, VT <: AbstractVector{T}}
    x = view(xr, 1:ncl.nx)
    r = view(xr, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.cons!(ncl.nlp, x, cx)
    axpy!(one(T), r, cx)
    return cx
end

function NLPModels.grad!(ncl::NCLModel{T, VT}, xr::VT, gxr::VT) where {T, VT <: AbstractVector{T}}
    sense = NLPModels.get_minimize(ncl.nlp) ? T(1) : T(-1)
    x = view(xr, 1:ncl.nx)
    r = view(xr, 1+ncl.nx:ncl.nx+ncl.nr)
    gx = view(gxr, 1:ncl.nx)
    gr = view(gxr, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.grad!(ncl.nlp, x, gx)
    gx .*= sense
    gr .= .-ncl.yk .+ ncl.ρk[] .* r
    return gxr
end

function NLPModels.jprod!(ncl::NCLModel{T, VT}, xr::VT, v::VT, Jv::VT) where {T, VT <: AbstractVector{T}}
    x = view(xr, 1:ncl.nx)
    vx = view(v, 1:ncl.nx)
    vr = view(v, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.jprod!(ncl.nlp, x, vx, Jv)
    axpy!(one(T), vr, Jv)
    return Jv
end

function NLPModels.jtprod!(ncl::NCLModel{T, VT}, xr::VT, v::VT, Jtv::VT) where {T, VT <: AbstractVector{T}}
    x = view(xr, 1:ncl.nx)
    Jtvx = view(Jtv, 1:ncl.nx)
    Jtvr = view(Jtv, 1+ncl.nx:ncl.nx+ncl.nr)
    NLPModels.jtprod!(ncl.nlp, x, v, Jtvx)
    Jtvr .= v
    return Jtv
end

function NLPModels.jac_structure!(ncl::NCLModel, jrows::VI, jcols::VI) where {VI <: AbstractVector{Int}}
    m, n = NLPModels.get_ncon(ncl), NLPModels.get_nvar(ncl)
    nnjx = NLPModels.get_nnzj(ncl.nlp)
    nnjxr = NLPModels.get_nnzj(ncl)
    jrowsx = view(jrows, 1:nnjx)
    jcolsx = view(jcols, 1:nnjx)
    NLPModels.jac_structure!(ncl.nlp, jrowsx, jcolsx)
    jrows[nnjx+1:nnjxr] .= 1:m
    jcols[nnjx+1:nnjxr] .= ncl.nx+1:n
    return (jrows, jcols)
end

function NLPModels.jac_coord!(ncl::NCLModel{T, VT}, xr::VT, jac::VT) where {T, VT <: AbstractVector{T}}
    nnjx = NLPModels.get_nnzj(ncl.nlp)
    nnjr = NLPModels.get_nnzj(ncl)
    x = view(xr, 1:ncl.nx)
    jacx = view(jac, 1:nnjx)
    NLPModels.jac_coord!(ncl.nlp, x, jacx)
    jac[nnjx+1:nnjr] .= one(T)
    return jac
end

function NLPModels.hess_structure!(ncl::NCLModel, hrows::VI, hcols::VI) where {VI <: AbstractVector{Int}}
    n = NLPModels.get_nvar(ncl)
    nnzhx = NLPModels.get_nnzh(ncl.nlp)
    nnzhxr = NLPModels.get_nnzh(ncl)
    hrowsx = view(hrows, 1:nnzhx)
    hcolsx = view(hcols, 1:nnzhx)
    NLPModels.hess_structure!(ncl.nlp, hrowsx, hcolsx)
    hrows[nnzhx+1:nnzhxr] .= ncl.nx+1:n
    hcols[nnzhx+1:nnzhxr] .= ncl.nx+1:n
    return (hrows, hcols)
end

function NLPModels.hess_coord!(
    ncl::NCLModel{T, VT},
    xr::AbstractVector{T},
    y::AbstractVector{T},
    hess::AbstractVector{T};
    obj_weight::T=one(T),
) where {T, VT <: AbstractVector{T}}
    sense = NLPModels.get_minimize(ncl.nlp) ? T(1) : T(-1)
    nnzhx = NLPModels.get_nnzh(ncl.nlp)
    nnzhxr = NLPModels.get_nnzh(ncl)
    x = view(xr, 1:ncl.nx)
    hessx = view(hess, 1:nnzhx)
    NLPModels.hess_coord!(ncl.nlp, x, y, hessx; obj_weight=sense * obj_weight)
    hess[nnzhx+1:nnzhxr] .= obj_weight * ncl.ρk[]
    return hess
end

