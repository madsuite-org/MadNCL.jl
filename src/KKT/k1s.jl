
"""
    K1sAuglagKKTSystem

Implement the KKT system (K1s) for the augmented Lagrangian algorithm NCL.
The blocks associated to the variables ``(r, s, y)`` are removed in the left-hand-side, reducing
the size of the KKT matrix.

The original augmented KKT system for augmented Lagrangian is:
```
[ H + Σₓ     0            Jᵀ] [Δx]     [ d1]
[   0     ρI + Σᵣ         I ] [Δr]  =  [ d2]
[   0        0      Σₛ   -I ] [Δs]     [ d3]
[   J        I      -I    0 ] [Δy]     [ d4]
```
By removing the block associated to ``(Δr, Δs, Δy)`` and setting ``Π := (ρI + Σᵣ)``,
we obtain the K1s formulation:
```
K Δx = d1 + Jᵀ (Π d4 - d2) + Jᵀ Π (Σₛ + Π)⁻¹ (d3 + d2 - Π d4)
```
with the condensed matrix ``K = H + Σₓ + Jᵀ Ω J`` depending
on the diagonal matrix ``Ω = Π Σₛ (Σₛ + Π)⁻¹``. We recover the remaining
descent direction as
```
Δs = (Σₛ + Π)⁻¹ (d3 + d2 - Π d4 + Π J Δx)
Δy = Π (J Δx - Δs - d4) + d2
Δr = Π⁻¹ (d2 - Δy)

```

"""
struct K1sAuglagKKTSystem{T, VT, MT, VI, VI32, LS, EXT, VD1, VD2} <: MadNLPCore.AbstractKKTSystem{T, VT, MT, MadNLPCore.ExactHessian{T, VT}}
    hess_callback::VT
    jac_callback::VT
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    pr_diag_reduced::VT
    buffer1::VT
    buffer2::VT
    buffer3::VT
    ρk::VT
    θk::VT
    # slack diagonal buffer
    diag_buffer::VT
    dptr::VD1
    hptr::VD1
    jptr::VD2
    # Condensed system
    aug_com::MT
    # Hessian
    hess_raw::MadNLPCore.SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::MadNLPCore.SparseMatrixCOO{T,Int32,VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_eq::VI
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    n::Int
    m::Int
    nx::Int
    nr::Int
    nnz_jac::Int
    nnz_hess::Int
    ext::EXT
    etc::Dict{Symbol, Any}
end

# Build KKT system directly from SparseCallback
function MadNLPCore.create_kkt_system(
    ::Type{K1sAuglagKKTSystem},
    cb::MadNLPCore.SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=MadNLPCore.default_options(linear_solver),
    hessian_approximation=MadNLPCore.ExactHessian,
    qn_options=MadNLPCore.QuasiNewtonOptions(),
) where {T,VT}
    nlp = cb.nlp
    nr = nlp.nr
    n = cb.nvar
    m = cb.ncon
    ind_ineq = cb.ind_ineq
    n_slack = length(ind_ineq)
    n_tot = n + n_slack - nr
    nx = n - nr

    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    VI = typeof(ind_ineq)
    ind_eq = if isa(ind_ineq, Vector)
        setdiff(1:m, ind_ineq)
    else
        ind_ineq_host = Vector(ind_ineq)
        VI(setdiff(1:m, ind_ineq_host))
    end

    # Evaluate sparsity pattern
    jac_sparsity_I = MadNLPCore.create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = MadNLPCore.create_array(cb, Int32, cb.nnzj)
    MadNLPCore._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    quasi_newton = MadNLPCore.create_quasi_newton(hessian_approximation, cb, n)
    hess_sparsity_I, hess_sparsity_J = MadNLPCore.build_hessian_structure(cb, hessian_approximation)
    MadNLPCore.force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    # Build augmented KKT system without regularized variable r
    pr_diag_reduced = VT(undef, n_tot)

    pr_diag = VT(undef, n_tot+nr)
    du_diag = VT(undef, m)
    reg = VT(undef, n_tot+nr)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    hess_callback = VT(undef, cb.nnzh)
    jac_callback = VT(undef, cb.nnzj)

    diag_buffer = VT(undef, m)
    buffer1 = VT(undef, n_tot)
    buffer2 = VT(undef, nr)
    buffer3 = VT(undef, nx)

    ρk = VT(undef, m)
    θk = VT(undef, m)
    fill!(ρk, zero(T))
    fill!(θk, zero(T))

    # COO matrices
    # Hessian
    nnz_hess = cb.nnzh - nr
    hess_x_I = hess_sparsity_I[1:cb.nnzh - nr]
    hess_x_J = hess_sparsity_J[1:cb.nnzh - nr]
    hess = VT(undef, nnz_hess)
    hess_raw = MadNLPCore.SparseMatrixCOO(
        nx, nx,
        hess_x_I,
        hess_x_J,
        hess,
    )
    hess_com, hess_csc_map = MadNLPCore.coo_to_csc(hess_raw)
    # Jacobian
    # N.B: we build the transpose of the Jacobian matrix as it
    #      is easier to assemble the condensed matrix afterwards.
    nnz_jac = cb.nnzj - nr
    jac = VT(undef, nnz_jac)
    jac_transpose_coo = MadNLPCore.SparseMatrixCOO(
        nx, m,
        jac_sparsity_J[1:nnz_jac],
        jac_sparsity_I[1:nnz_jac],
        jac,
    )
    jt_csc, jt_csc_map = MadNLPCore.coo_to_csc(jac_transpose_coo)

    # Symbolic analysis for condensed system.
    aug_com, dptr, hptr, jptr = MadNLPCore.build_condensed_aug_symbolic(
            hess_com,
            jt_csc
    )

    # The condensed matrix is definite: no dual block.
    MadNLPCore.set_block_structure!(opt_linear_solver, size(aug_com, 1), 0)
    _linear_solver = MadNLPCore.create_linear_solver(
        linear_solver,
        aug_com; opt = opt_linear_solver
    )

    ext = MadNLPCore.get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    etc = Dict{Symbol, Any}()
    return K1sAuglagKKTSystem(
            hess_callback, jac_callback,
            reg, pr_diag, du_diag,
            l_diag, u_diag, l_lower, u_lower,
            pr_diag_reduced,
            buffer1, buffer2, buffer3,
            ρk, θk,
            diag_buffer, dptr, hptr, jptr,
            aug_com,
            hess_raw, hess_com, hess_csc_map,
            jac_transpose_coo, jt_csc, jt_csc_map,
            _linear_solver,
            ind_eq, ind_ineq, cb.ind_lb, cb.ind_ub,
            n_tot, m, nx, nr, nnz_jac, nnz_hess,
            ext, etc,
    )
end

MadNLPCore.num_variables(kkt::K1sAuglagKKTSystem) = length(kkt.pr_diag)
MadNLPCore.get_jacobian(kkt::K1sAuglagKKTSystem) = kkt.jac_callback
MadNLPCore.get_hessian(kkt::K1sAuglagKKTSystem) = kkt.hess_callback

function MadNLPCore.is_inertia_correct(kkt::K1sAuglagKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
end

MadNLPCore.block_structure(kkt::K1sAuglagKKTSystem) = (size(kkt.aug_com, 1), 0)

function MadNLPCore.initialize!(kkt::K1sAuglagKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess_callback, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

function MadNLPCore.compress_jacobian!(kkt::K1sAuglagKKTSystem)
    ns = length(kkt.ind_ineq)
    # Update values in COO matrix
    kkt.jac_raw.V[1:kkt.nnz_jac] .= @view kkt.jac_callback[1:kkt.nnz_jac]
    # Transfer to CSC
    MadNLPCore.transfer!(kkt.jt_csc, kkt.jac_raw, kkt.jt_csc_map)
end

function MadNLPCore.compress_hessian!(kkt::K1sAuglagKKTSystem)
    nr = kkt.nr
    # Update values in COO matrix
    kkt.hess_raw.V[1:kkt.nnz_hess] .= @view kkt.hess_callback[1:kkt.nnz_hess]
    # Update ρ
    kkt.ρk .= @view kkt.hess_callback[kkt.nnz_hess+1:kkt.nnz_hess+nr]
    # Transfer to CSC
    MadNLPCore.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLPCore.jtprod!(y::AbstractVector, kkt::K1sAuglagKKTSystem, x::AbstractVector)
    nx, nr, ns = kkt.nx, kkt.nr, length(kkt.ind_ineq)
    jtx = view(kkt.buffer1, 1:nx)
    mul!(jtx, kkt.jt_csc, x)
    # Unpack results
    # Variable
    y[1:nx] .= jtx
    # Regularization
    y[nx+1:nx+nr] .= x
    # Slack
    x_ineq = view(x, kkt.ind_ineq)
    y[nx+nr+1:nx+nr+ns] .= .-x_ineq
    return y
end

function MadNLPCore.build_condensed_aug_coord!(kkt::K1sAuglagKKTSystem{T,VT,MT}) where {T, VT, MT <: SparseMatrixCSC{T}}
    MadNLPCore._build_condensed_aug_coord!(
        kkt.aug_com, kkt.pr_diag_reduced, kkt.hess_com, kkt.jt_csc, kkt.diag_buffer,
        kkt.dptr, kkt.hptr, kkt.jptr
    )
end

function MadNLPCore.build_kkt!(kkt::K1sAuglagKKTSystem)
    nx, nr, ns = kkt.nx, kkt.nr, length(kkt.ind_ineq)
    Σx = @view kkt.pr_diag[1:nx]
    Σr = @view kkt.pr_diag[nx+1:nx+nr]
    Σs = @view kkt.pr_diag[nx+nr+1:nx+nr+ns]
    kkt.θk .= (kkt.ρk .+ Σr)

    kkt.pr_diag_reduced[1:nx] .= Σx
    kkt.pr_diag_reduced[nx+1:nx+ns] .= Σs

    kkt.diag_buffer[kkt.ind_eq] .= kkt.θk[kkt.ind_eq]
    kkt.diag_buffer[kkt.ind_ineq] .= kkt.θk[kkt.ind_ineq] .* Σs ./ (Σs .+ kkt.θk[kkt.ind_ineq])

    MadNLPCore.build_condensed_aug_coord!(kkt)
end

function MadNLPCore.solve_kkt!(kkt::K1sAuglagKKTSystem{T}, w::MadNLPCore.AbstractKKTVector) where T
    MadNLPCore.reduce_rhs!(w.xp_lr, MadNLPCore.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLPCore.dual_ub(w), kkt.u_diag)

    nx, nr, ns = kkt.nx, kkt.nr, length(kkt.ind_ineq)
    n, m = kkt.n, kkt.m

    Σs = @view kkt.pr_diag[nx+nr+1:nx+nr+ns]
    θk = kkt.θk
    ds = view(kkt.buffer1, nx+1:nx+ns)
    dy = kkt.buffer2           # size (m)
    dx = kkt.buffer3           # size (nx)

    # Unpack right-hand-side
    w_ = MadNLPCore.full(w)
    wx = view(w_, 1:nx)
    wr = view(w_, nx+1:nx+nr)
    ws = view(w_, nx+nr+1:nx+nr+ns)
    wy = view(w_, nx+nr+ns+1:nx+nr+ns+m)

    # Remove Δr
    dy .= kkt.θk .* (wy .- wr ./ θk)
    # Remove Δy
    mul!(dx, kkt.jt_csc, dy)
    dx .= dx .+ wx
    ds .= ws .- dy[kkt.ind_ineq]
    # Remove Δs
    fill!(dy, zero(T))
    dy[kkt.ind_ineq] .= (kkt.θk[kkt.ind_ineq] .* ds) ./ (Σs .+ kkt.θk[kkt.ind_ineq])
    mul!(dx, kkt.jt_csc, dy, one(T), one(T))

    MadNLPCore.solve_linear_system!(kkt.linear_solver, dx)

    # Unpack solution
    # Δx
    wx .= dx
    # Δs
    mul!(dy, kkt.jt_csc', wx)
    ws .= (ds .+ kkt.θk[kkt.ind_ineq] .* dy[kkt.ind_ineq]) ./ (Σs + kkt.θk[kkt.ind_ineq])
    # Δy
    dy[kkt.ind_ineq] .-= ws
    wy .= kkt.θk .* (dy .- wy) .+ wr
    # Δr
    wr .= (wr .- wy) ./ θk

    MadNLPCore.finish_aug_solve!(kkt, w)
    return w
end

function mul!(w::MadNLPCore.AbstractKKTVector{T}, kkt::K1sAuglagKKTSystem, v::MadNLPCore.AbstractKKTVector, alpha = one(T), beta = zero(T)) where {T}
    n, m = kkt.n, kkt.m
    nx, nr, ns = kkt.nx, kkt.nr, length(kkt.ind_ineq)
    ρk = kkt.ρk
    # Unpack vector x
    vx = view(MadNLPCore.full(v), 1:nx)
    vr = view(MadNLPCore.full(v), nx+1:nx+nr)
    vs = view(MadNLPCore.full(v), nx+nr+1:nx+nr+ns)
    vy = view(MadNLPCore.full(v), nx+nr+ns+1:nx+nr+ns+m)
    # Unpack vector w
    wx = view(MadNLPCore.full(w), 1:nx)
    wr = view(MadNLPCore.full(w), nx+1:nx+nr)
    ws = view(MadNLPCore.full(w), nx+nr+1:nx+nr+ns)
    wy = view(MadNLPCore.full(w), nx+nr+ns+1:nx+nr+ns+m)

    wy_ineq = view(wy, kkt.ind_ineq)
    vy_ineq = view(vy, kkt.ind_ineq)

    symul!(wx, kkt.hess_com, vx, alpha, beta)
    mul!(wx, kkt.jt_csc, vy, alpha, one(T))

    ws .= beta .* ws .- alpha .* vy_ineq
    wr .= beta .* wr .+ alpha .* (ρk .* vr .+ vy)

    mul!(wy, kkt.jt_csc', vx, alpha, beta)
    wy_ineq .-= alpha .* vs
    wy .+= alpha .* vr

    MadNLPCore._kktmul!(w,v,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end

