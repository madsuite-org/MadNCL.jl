"""
    ScaledK2rKKTSystem

Implement the KKT system (K2r) for the augmented Lagrangian algorithm NCL.
The blocks associated to the variables ``r`` are removed in the left-hand-side, reducing
the size of the KKT matrix.

The original augmented KKT system for augmented Lagrangian is:
```
[ H + Σₓ     0            Jᵀ] [Δx]     [ d1]
[   0     ρI + Σᵣ         I ] [Δr]  =  [ d2]
[   0        0      Σₛ   -I ] [Δs]     [ d3]
[   J        I      -I    0 ] [Δy]     [ d4]
```
By removing the block associated to ``Δr``and setting ``θ := (ρI + Σᵣ)⁻¹``,
we obtain the K2 formulation:
```
[ H + Σₓ            Jᵀ] [Δx]     [ d1]
[   0      Σₛ      -I ] [Δs]  =  [ d3]
[   J      -I     -θI ] [Δy]     [ d4 - θ d2]
```
and we recover ``Δr = θ * (d2 - Δy)``.

"""
struct ScaledK2rKKTSystem{T, VT, MT, VI, VI32, LS} <: MadNLP.AbstractKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    nlp::NCLModel{T, VT}
    hess_callback::VT
    jac_callback::VT
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    scaling_factor::VT
    pr_diag_reduced::VT
    du_diag_reduced::VT
    buffer1::VT
    buffer2::VT
    buffer3::VT
    buffer4::VT
    buffer5::VT
    ρk::VT
    θk::VT
    # Augmented system
    aug_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    scaled_aug_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    # Hessian
    hess_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    ind_fixed::VI
    n::Int
    m::Int
    nnz_jac::Int
    nnz_hess::Int
end

# Build KKT system directly from SparseCallback
function MadNLP.create_kkt_system(
    ::Type{ScaledK2rKKTSystem},
    cb::MadNLP.SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
) where {T,VT}
    nlp = cb.nlp
    nx, nr = nlp.nx, nlp.nr
    n = cb.nvar
    m = cb.ncon
    n_slack = length(ind_cons.ind_ineq)
    n_tot = nx + n_slack
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    n_fixed = length(ind_cons.ind_fixed)
    ind_ineq = ind_cons.ind_ineq

    # Evaluate sparsity pattern
    jac_sparsity_I = MadNLP.create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = MadNLP.create_quasi_newton(hessian_approximation, cb, n)
    hess_sparsity_I, hess_sparsity_J = MadNLP.build_hessian_structure(cb, hessian_approximation)
    MadNLP.force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    # Remove regularized variable r from sparsity pattern
    nnz_jac = cb.nnzj - nr
    nnz_hess = cb.nnzh - nr
    # Extract indexes associated to x
    hess_x_I = hess_sparsity_I[1:cb.nnzh - nr - n_fixed]
    hess_x_J = hess_sparsity_J[1:cb.nnzh - nr - n_fixed]
    # N.B. Special treatment for fixed variables in MadNLP: the nnz entries
    # for the fixed variables are moved at the end.
    if n_fixed > 0
        append!(hess_x_I, hess_sparsity_I[cb.nnzh-n_fixed+1:cb.nnzh])
        append!(hess_x_J, hess_sparsity_J[cb.nnzh-n_fixed+1:cb.nnzh])
    end

    n_aug_kkt = n_tot+m
    nnz_aug_kkt = nx+n_slack+m+nnz_hess+nnz_jac+n_slack

    I = MadNLP.create_array(cb, Int32, nnz_aug_kkt)
    J = MadNLP.create_array(cb, Int32, nnz_aug_kkt)
    V = VT(undef, nnz_aug_kkt)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n_tot+nnz_jac+n_slack+nnz_hess+m

    index_reg = 1:n_tot
    index_hess = n_tot+1:n_tot+nnz_hess
    index_jacv = n_tot+nnz_hess+1:n_tot+nnz_hess+nnz_jac
    index_jacs = n_tot+nnz_hess+nnz_jac+1:n_tot+nnz_hess+nnz_jac+n_slack
    index_du = n_tot+nnz_hess+nnz_jac+n_slack+1:offset

    # Build augmented KKT system without regularized variable r
    I[index_reg] .= 1:n_tot
    I[index_hess] .= hess_x_I
    I[index_jacv] .= jac_sparsity_I[1:nnz_jac] .+ n_tot
    I[index_jacs] .= ind_ineq .+ n_tot
    I[index_du] .= (n_tot+1:n_tot+m)

    J[index_reg] .= 1:n_tot
    J[index_hess] .= hess_x_J
    J[index_jacv] .= jac_sparsity_J[1:nnz_jac]
    J[index_jacs] .= (nx+1:nx+n_slack)
    J[index_du] .= (n_tot+1:n_tot+m)

    pr_diag_reduced = MadNLP._madnlp_unsafe_wrap(V, n_tot)
    du_diag_reduced = MadNLP._madnlp_unsafe_wrap(V, m, nnz_jac+n_slack+nnz_hess+n_tot+1)

    pr_diag = VT(undef, n_tot+nr)
    du_diag = VT(undef, m)
    reg = VT(undef, n_tot+nr)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    # Scaling edits
    scaling_factor = VT(undef, n_tot)
    buffer4 = VT(undef, n_tot)
    buffer5 = VT(undef, n_tot)

    hess_callback = VT(undef, cb.nnzh)
    jac_callback = VT(undef, cb.nnzj)

    buffer1 = VT(undef, n_tot)
    buffer2 = VT(undef, nr)
    buffer3 = VT(undef, n_tot + m)

    ρk = VT(undef, m)
    θk = VT(undef, m)
    fill!(ρk, zero(T))
    fill!(θk, zero(T))

    # COO matrices
    # Augmented system
    aug_raw = MadNLP.SparseMatrixCOO(n_aug_kkt, n_aug_kkt, I, J, V)
    # Jacobian
    jac = MadNLP._madnlp_unsafe_wrap(V, nnz_jac+n_slack, nnz_hess+n_tot+1)
    jac_raw = MadNLP.SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I[1:nnz_jac]; ind_ineq],
        Int32[jac_sparsity_J[1:nnz_jac]; nx+1:nx+n_slack],
        jac,
    )
    # Hessian
    hess = MadNLP._madnlp_unsafe_wrap(V, nnz_hess, n_tot+1)
    hess_raw = MadNLP.SparseMatrixCOO(
        n_tot, n_tot,
        hess_x_I,
        hess_x_J,
        hess,
    )

    aug_vec_length = n_tot+m
    scaled_aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,copy(V))

    aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)
    jac_com, jac_csc_map = MadNLP.coo_to_csc(jac_raw)
    hess_com, hess_csc_map = MadNLP.coo_to_csc(hess_raw)

    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver
    )

    return ScaledK2rKKTSystem(
        nlp,
        hess_callback, jac_callback,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower, 
        scaling_factor,
        pr_diag_reduced,
        du_diag_reduced,
        buffer1, buffer2, buffer3, buffer4, buffer5,
        ρk, θk,
        aug_raw, aug_com, aug_csc_map,
        scaled_aug_raw,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub, ind_cons.ind_fixed,
        n_tot, m, nnz_jac, nnz_hess,
    )

end

MadNLP.num_variables(kkt::ScaledK2rKKTSystem) = length(kkt.pr_diag)
MadNLP.get_jacobian(kkt::ScaledK2rKKTSystem) = kkt.jac_callback
MadNLP.get_hessian(kkt::ScaledK2rKKTSystem) = kkt.hess_callback

function MadNLP.is_inertia_correct(kkt::ScaledK2rKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == kkt.n)
end

function MadNLP.initialize!(kkt::ScaledK2rKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess_callback, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(kkt.scaling_factor, one(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

function MadNLP.compress_jacobian!(kkt::ScaledK2rKKTSystem)
    ns = length(kkt.ind_ineq)
    # Update values in COO matrix
    kkt.jac_raw.V[1:kkt.nnz_jac] .= @view kkt.jac_callback[1:kkt.nnz_jac]
    kkt.jac_raw.V[end-ns+1:end] .= -1.0
    # Transfer to CSC
    MadNLP.transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function MadNLP.compress_hessian!(kkt::ScaledK2rKKTSystem)
    nr = kkt.nlp.nr
    n_fixed = length(kkt.ind_fixed)
    # Update values in COO matrix
    kkt.hess_raw.V[1:kkt.nnz_hess-n_fixed] .= @view kkt.hess_callback[1:kkt.nnz_hess-n_fixed]
    # Update ρ
    kkt.ρk .= @view kkt.hess_callback[kkt.nnz_hess-n_fixed+1:kkt.nnz_hess+nr-n_fixed]
    # Update fixed values
    if n_fixed > 0
        kkt.hess_raw.V[kkt.nnz_hess-n_fixed+1:kkt.nnz_hess] .= @view kkt.hess_callback[kkt.nnz_hess+nr-n_fixed+1:kkt.nnz_hess+nr]
    end
    # Transfer to CSC
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.jtprod!(y::AbstractVector, kkt::ScaledK2rKKTSystem, x::AbstractVector)
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    mul!(kkt.buffer1, kkt.jac_com', x)
    # Unpack results
    y[1:nx] .= @view kkt.buffer1[1:nx]
    y[nx+1:nx+nr] .= x
    y[nx+nr+1:nx+nr+ns] .= @view kkt.buffer1[nx+1:nx+ns]
    return y
end

function MadNLP.build_kkt!(kkt::ScaledK2rKKTSystem)
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    m, n = size(kkt.jac_raw)
    Σx = @view kkt.pr_diag[1:nx]
    Σr = @view kkt.pr_diag[nx+1:nx+nr]
    Σs = @view kkt.pr_diag[nx+nr+1:nx+nr+ns]
    kkt.θk .= 1.0 ./ (kkt.ρk .+ Σr)
    kkt.du_diag_reduced .= .-kkt.θk
    kkt.pr_diag_reduced[1:nx] .= Σx
    kkt.pr_diag_reduced[nx+1:nx+ns] .= Σs
    MadNLP._build_scale_augmented_system_coo!(
        kkt.scaled_aug_raw,
        kkt.aug_raw,
        kkt.scaling_factor,
        n, m,
    )
    MadNLP.transfer!(kkt.aug_com, kkt.scaled_aug_raw, kkt.aug_csc_map)
end

function MadNLP.solve!(kkt::ScaledK2rKKTSystem, w::MadNLP.AbstractKKTVector)
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)

    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    n, m = kkt.n, kkt.m

    θk = kkt.θk
    d = kkt.buffer3

    # Unpack right-hand-side
    w_ = MadNLP.full(w)
    wx = view(w_, 1:nx)
    wr = view(w_, nx+1:nx+nr)
    ws = view(w_, nx+nr+1:nx+nr+ns)
    wy = view(w_, nx+nr+ns+1:nx+nr+ns+m)

    d[1:nx] .= wx
    d[nx+1:nx+ns] .= ws
    d[nx+ns+1:nx+ns+m] .= wy .- θk .* wr

    MadNLP.solve!(kkt.linear_solver, d)

    # Unpack solution
    copyto!(wx, 1, d, 1, nx)
    copyto!(ws, 1, d, nx+1, ns)
    copyto!(wy, 1, d, nx+ns+1, m)
    wr .= θk .* (wr .- wy)

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

function mul!(w::MadNLP.AbstractKKTVector{T}, kkt::ScaledK2rKKTSystem, v::MadNLP.AbstractKKTVector, alpha = one(T), beta = zero(T)) where {T}
    n, m = kkt.n, kkt.m
    nx, nr, ns = kkt.nlp.nx, kkt.nlp.nr, length(kkt.ind_ineq)
    ρk = kkt.ρk
    # Unpack vector x
    vx = view(MadNLP.full(v), 1:nx)
    vr = view(MadNLP.full(v), nx+1:nx+nr)
    vs = view(MadNLP.full(v), nx+nr+1:nx+nr+ns)
    vy = view(MadNLP.full(v), nx+nr+ns+1:nx+nr+ns+m)
    # Unpack vector w
    wx = view(MadNLP.full(w), 1:nx)
    wr = view(MadNLP.full(w), nx+1:nx+nr)
    ws = view(MadNLP.full(w), nx+nr+1:nx+nr+ns)
    wy = view(MadNLP.full(w), nx+nr+ns+1:nx+nr+ns+m)

    # Load buffers to work in (x, s) space instead of (x, r, s)
    yp = kkt.buffer1
    yp[1:nx] .= beta .* wx
    yp[nx+1:nx+ns] .= beta .* ws
    xp = view(kkt.buffer3, 1:nx+ns)
    xp[1:nx] .= vx
    xp[nx+1:nx+ns] .= vs

    symul!(yp, kkt.hess_com, xp, alpha, one(T))
    mul!(yp, kkt.jac_com', vy, alpha, one(T))

    wx .= @view yp[1:nx]
    ws .= @view yp[nx+1:nx+ns]
    wr .= beta .* wr .+ alpha .* (ρk .* vr .+ vy)

    mul!(wy, kkt.jac_com, xp, alpha, beta)
    wy .+= alpha .* vr

    MadNLP._kktmul!(w,v,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end

