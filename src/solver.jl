
import MadNLP: full, primal, dual, dual_lb, dual_ub

#=
    NCLOptions
=#

@kwdef struct NCLOptions{T}
    verbose::Bool = true
    extrapolation::Bool = true
    extrapolation_rho::T = T(0.3)
    scaling::Bool = true
    scaling_max_gradient::T = T(1)
    opt_tol::T = T(1e-6)
    feas_tol::T = T(1e-6)
    constr_viol_tol::T = max(T(100) * feas_tol, T(1e-4))
    rho_init::T = T(1e2)
    rho_max::T = T(1e12)
    max_auglag_iter::Int = 20
    mu_init::T = T(1e-1)
    mu_tau::T = T(1.99)
    mu_fac::T = T(0.2)
    mu_min::T = T(1e-9)
end

#=
    NCLSolver
=#

struct NCLSolver{T, VT, M}
    ncl::NCLModel{T, VT, M}
    ipm::MadNLP.MadNLPSolver{T, VT}
    options::NCLOptions{T}
    n::Int
    m::Int
end

function NCLSolver(nlp::NLPModels.AbstractNLPModel{T, VT}; ncl_options=NCLOptions{T}(), ipm_options...) where {T, VT}
    ncl = if ncl_options.scaling
        NCLModel(ScaledModel(nlp; max_gradient=ncl_options.scaling_max_gradient))
    else
        NCLModel(nlp)
    end
    solver = MadNLP.MadNLPSolver(
        ncl;
        nlp_scaling=false,
        ipm_options...,
    )
    m = MadNLP.n_constraints(solver.cb)
    # Remove regularization variables to work in the original space
    n = MadNLP.n_variables(solver.cb) - m
    @assert solver.m == m
    return NCLSolver{T, VT, typeof(ncl.nlp)}(ncl, solver, ncl_options, n, m)
end

#=
    NCLStats
=#

mutable struct NCLStats{T, VT} <: AbstractExecutionStats
    status::MadNLP.Status
    solution::VT
    regularization::VT
    objective::T
    dual_feas::T
    primal_feas::T
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    iter::Int
    counters::MadNLP.MadNLPCounters
end

function NCLStats(solver::NCLSolver{T, VT, M}, status) where {T, VT, M<:NLPModels.AbstractNLPModel}
    ipm = solver.ipm # MadNLP instance
    ncl = solver.ncl
    # Get original number of variables (before removing the fixed variables)
    n = NLPModels.get_nvar(ncl.nlp)
    m = solver.m
    is_min = NLPModels.get_minimize(ncl.nlp)
    x = similar(VT, n + m)
    zl = similar(VT, n + m)
    zu = similar(VT, n + m)
    MadNLP.unpack_x!(x, ipm.cb, MadNLP.variable(solver.ipm.x))
    MadNLP.unpack_z!(zl, ipm.cb, MadNLP.variable(solver.ipm.zl))
    MadNLP.unpack_z!(zu, ipm.cb, MadNLP.variable(solver.ipm.zu))

    y = copy(solver.ipm.y)
    r = x[n+1:n+m]
    # Evaluate the objective directly, as yk and ρk may have changed
    # since the last evaluation of ipm.obj_val.
    obj_val = NLPModels.obj(ncl.nlp, x[1:n])
    MadNLP.update_z!(ipm.cb, x, y, zl, zu, ipm.jacl)
    # Scale back problem data
    if isa(ncl.nlp, ScaledModel)
        obj_scale, con_scale = ncl.nlp.scaling_obj, ncl.nlp.scaling_cons
        obj_val /= obj_scale
        y .= y .* con_scale ./ obj_scale
        zl .= zl ./ obj_scale
        zu .= zu ./ obj_scale
    end
    return NCLStats{T, VT}(
        status,
        x[1:n],
        r,
        obj_val,
        solver.ipm.inf_du,
        norm(r, Inf),
        is_min ? y : .-y,
        zl[1:n],
        zu[1:n],
        solver.ipm.cnt.k,
        solver.ipm.cnt,
    )
end

function getStatus(result::NCLStats)
    if result.status == MadNLP.SOLVE_SUCCEEDED
        println("Optimal solution found.")
    elseif result.status == MadNLP.INFEASIBLE_PROBLEM_DETECTED
        println("Convergence to an infeasible point.")
    elseif result.status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        println("Maximum number of iterations reached.")
    else
        println("Unknown return status.")
    end
end

#=
    NCL Algorithm
=#

function _introduce(nx, nr)
    println("MadNCL algorithm\n")

    println("Total number of variables............................:      ", nx)
    println("Total number of constraints..........................:      ", nr)
    println()
end

function _log_header()
    @printf(
        "outer  inner     objective    inf_pr   inf_du    η        μ       ρ \n"
    )
end

function _log_iter(nit, flag, n_inner, obj, inf_pr, inf_du, alpha, mu, rho)
    @printf(
        "%5s%1s %5i %+13.7e %6.2e %6.2e %6.2e %6.1e %6.2e\n",
        nit, flag, n_inner, obj, inf_pr, inf_du, alpha, mu, rho,
    )
end

function get_inf_du(solver::NCLSolver)
    return solver.ipm.inf_du
end

function get_constr_viol_tol(ncl::NCLModel, r::AbstractVector)
    return norm(r, Inf)
end
function get_constr_viol_tol(ncl::NCLModel{T, VT, M}, r::AbstractVector) where {T, VT, M<:ScaledModel}
    m = NLPModels.get_ncon(ncl)
    if m > 0
        con_scale = ncl.nlp.scaling_cons
        # Compute norm-Inf with mapreduce
        return mapreduce((x, c) -> abs(x) / c, max, r, con_scale)
    else
        return 0.0
    end
end

function setup!(solver::MadNLP.MadNLPSolver{T}; μ=1e-1, tol=1e-8) where T
    barrier = solver.opt.barrier
    # Update options
    barrier.mu_init = μ
    solver.opt.tol = tol
    solver.mu = barrier.mu_init
    # Ensure the barrier parameter is fixed
    barrier.mu_min = barrier.mu_init

    # Refresh values
    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)

    # Update filter
    theta = MadNLP.get_theta(solver.c)
    solver.theta_max = T(1e4) * max(1,theta)
    solver.theta_min = T(1e-4) * max(1,theta)
    solver.tau = max(solver.opt.tau_min,one(T)-barrier.mu_init)
    empty!(solver.filter)
    push!(solver.filter, (solver.theta_max,-Inf))

    return MadNLP.REGULAR
end

# RHS for extrapolation step
function set_aug_rhs_extrapolation!(solver::NCLSolver, kkt::MadNLP.AbstractKKTSystem, μ)
    ipm = solver.ipm
    ρ = solver.ncl.ρk[]

    n_ineq = length(ipm.ind_ineq)
    n, m = solver.n, solver.m

    f = primal(ipm.f)
    zl = full(ipm.zl)
    zu = full(ipm.zu)

    # Variables
    px = @view primal(ipm.p)[1:n]
    fx = @view primal(ipm.f)[1:n]
    jaclx = @view ipm.jacl[1:n]
    zlx = @view primal(ipm.zl)[1:n]
    zux = @view primal(ipm.zu)[1:n]
    # Slacks
    ps = @view primal(ipm.p)[n+m+1:n+m+n_ineq]
    fs = @view primal(ipm.f)[n+m+1:n+m+n_ineq]
    jacls = @view ipm.jacl[n+m+1:n+m+n_ineq]
    zls = @view primal(ipm.zl)[n+m+1:n+m+n_ineq]
    zus = @view primal(ipm.zu)[n+m+1:n+m+n_ineq]
    # Regularization
    r = @view primal(ipm.x)[n+1:n+m]
    pr = @view primal(ipm.p)[n+1:n+m]

    py = dual(ipm.p)
    pzl = dual_lb(ipm.p)
    pzu = dual_ub(ipm.p)

    px .= .-fx .+ zlx .- zux .- jaclx
    pr .= .-ρ .* r
    ps .= .-fs .+ zls .- zus .- jacls
    py .= .-ipm.c
    pzl .= (ipm.xl_r .- ipm.x_lr) .* ipm.zl_r .+ μ
    pzu .= (ipm.xu_r .- ipm.x_ur) .* ipm.zu_r .- μ
    return
end


# N.B.: The extrapolation step is described in detail in:
# [1, Section 3.2] Armand, Paul, and Riadh Omheni.
#    "A mixed logarithmic barrier-augmented Lagrangian method for nonlinear optimization."
#    Journal of Optimization Theory and Applications 173.2 (2017): 523-547.
# [2, Algorithm 1] Armand, Paul, Joël Benoist, and Dominique Orban.
#    "From global to local convergence of interior methods for nonlinear optimization."
#    Optimization Methods and Software 28.5 (2013): 1051-1080.
function extrapolation!(solver::NCLSolver{T}) where T
    ipm = solver.ipm
    ρ = solver.ncl.ρk[]
    rho = solver.options.extrapolation_rho
    μk = ipm.mu
    μ_fac, μ_tau, μ_min = solver.options.mu_fac, solver.options.mu_tau, solver.options.mu_min

    # Evaluate model with new rho and mu.
    ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
    MadNLP.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
    MadNLP.eval_cons_wrapper!(ipm, ipm.c, ipm.x)
    MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
    MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
    MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)

    # Previous KKT residual is the norm of the RHS
    set_aug_rhs_extrapolation!(solver, ipm.kkt, μk)
    res_p = norm(ipm.p.values, Inf)

    # Update barrier for extrapolation step
    μp = max(min(μk^μ_tau, μ_fac * μk), μ_min)
    τ = max(T(0.995), one(T) - μp)

    # Solve KKT system
    MadNLP.set_aug_diagonal!(ipm.kkt, ipm)
    set_aug_rhs_extrapolation!(solver, ipm.kkt, μp)
    is_solved = MadNLP.inertia_correction!(ipm.inertia_corrector, ipm)

    # If the linear solver has failed, we leave the extrapolation step
    if !is_solved
        return (false, res_p)
    end

    # Fraction to boundary
    alpha_p = MadNLP.get_alpha_max(
        primal(ipm.x),
        primal(ipm.xl),
        primal(ipm.xu),
        primal(ipm.d),
        τ,
    )
    alpha_d = MadNLP.get_alpha_z(
        ipm.zl_r,
        ipm.zu_r,
        dual_lb(ipm.d),
        dual_ub(ipm.d),
        τ,
    )

    # Take full Newton step
    axpy!(alpha_p, primal(ipm.d), primal(ipm.x))
    axpy!(alpha_p, dual(ipm.d), ipm.y)
    ipm.zl_r .+= alpha_d .* dual_lb(ipm.d)
    ipm.zu_r .+= alpha_d .* dual_ub(ipm.d)

    # Update barrier term
    ipm.mu = μk + alpha_p * (μp - μk)

    # Update callbacks
    ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
    MadNLP.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
    MadNLP.eval_cons_wrapper!(ipm, ipm.c, ipm.x)
    MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
    MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)

    # Compute KKT residual to check if crossover has succeeded
    set_aug_rhs_extrapolation!(solver, ipm.kkt, ipm.mu)
    res_k = norm(ipm.p.values, Inf)

    if res_k <= rho * res_p + T(10) * alpha_p^T(0.2) * μk
        return (true, res_k)
    else
        # If decrease is not sufficient, the solver does not take
        # the full step and we return to the previous iterate.
        ipm.mu = μk
        axpy!(-alpha_p, primal(ipm.d), primal(ipm.x))
        axpy!(-alpha_p, dual(ipm.d), ipm.y)
        ipm.zl_r .-= alpha_d .* dual_lb(ipm.d)
        ipm.zu_r .-= alpha_d .* dual_ub(ipm.d)
        # Refresh values in callback
        ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
        MadNLP.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
        MadNLP.eval_cons_wrapper!(ipm, ipm.c, ipm.x)
        return (false, res_p)
    end
end

function solve!(solver::NCLSolver{T}) where T
    n, m = solver.n, solver.m
    ncl = solver.ncl
    ipm = solver.ipm
    options = solver.options

    options.verbose && _introduce(n, m)

    # Parameters
    ### Penalty ρ
    ncl.ρk[] = options.rho_init
    ρ_max = options.rho_max
    tau_ρ = T(10)
    ### Barrier parameters
    μ = options.mu_init
    μ_min = options.mu_min
    μ_fac = options.mu_fac
    τ = options.mu_tau
    ### Forcing parameters
    γ = T(0.05)
    eps_c = T(10)*μ
    eps_d = T(100)*μ^(one(T)+γ)
    η = T(0.1)   # initial primal feasibility tolerance

    start_time = time()

    ncl_status = MadNLP.INITIAL

    # Unless the parameter `dual_initialized` is set to `true`,
    # MadNLP computes the initial multipliers y0 in `initialize!`.
    MadNLP.initialize!(ipm)
    # Update initial multiplier using multiplier computed by MadNLP in `initialize!`.
    ncl.yk .= ipm.y

    ipm.status = MadNLP.REGULAR
    cnt_it_inner = 0

    x = view(MadNLP.primal(ipm.x), 1:n)
    r = view(MadNLP.primal(ipm.x), 1+n:n+m)

    pr_feas = ipm.inf_pr
    du_feas = get_inf_du(solver)

    is_extrapolated = false
    options.verbose && _log_header()
    flag = " "
    options.verbose && _log_iter(0, flag, 0, ipm.obj_val, pr_feas, du_feas, η, μ, ncl.ρk[])

    iter = 1
    while iter <= options.max_auglag_iter
        # Update parameters in MadNLP
        setup!(
            ipm;
            tol=eps_d,
            μ=μ,
        )

        if iter >= 2 && options.extrapolation
            is_extrapolated, delta = extrapolation!(solver)
            flag = is_extrapolated ? "+" : " "
        end
        if !is_extrapolated
            status = MadNLP.regular!(ipm)
            if status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
                ncl_status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
                break
            end
            has_converged = status == MadNLP.SOLVE_SUCCEEDED
            flag = has_converged ? " " : "r"
        end

        # Get KKT residuals for scaled problem
        pr_feas = norm(r, Inf)
        du_feas = MadNLP.get_inf_du(
            MadNLP.full(ipm.f), MadNLP.full(ipm.zl), MadNLP.full(ipm.zu), ipm.jacl, 1.0,
        )
        cc_feas = MadNLP.get_inf_compl(
            ipm.x_lr, ipm.xl_r, ipm.zl_r, ipm.xu_r, ipm.x_ur, ipm.zu_r, ipm.mu, 1.0,
        )

        # Objective without the augmented Lagrangian terms.
        # N.B.: must be computed before yk and ρk are updated.
        obj_val = ipm.obj_val + dot(ncl.yk, r) - ncl.ρk[] * dot(r, r) / T(2)

        # Update parameters
        if is_extrapolated
            ncl.yk .= ipm.y
            ncl.ρk[] = min(max(one(T) / delta, T(1.1) * ncl.ρk[]), ρ_max)
            μ = ipm.mu  # mu has been updated previously when computing extrapolation step
        elseif pr_feas <= max(η, options.feas_tol) && has_converged
            ncl.yk .= ipm.y
            μ = max(min(μ^τ, μ_fac * μ), μ_min)
        else
            ncl.ρk[] = min(ncl.ρk[] * tau_ρ, ρ_max)
        end

        η = min(μ^T(1.1), T(0.1) * μ)
        eps_d = T(100) * μ^(one(T)+γ)
        eps_c = T(10) * μ

        # Log evolution
        ipm_iter = ipm.cnt.k
        options.verbose && _log_iter(iter, flag, ipm_iter, obj_val, pr_feas, du_feas, η, μ, ncl.ρk[])

        # Check convergence
        if (
            max(du_feas, cc_feas) <= options.opt_tol &&
            pr_feas <= options.feas_tol &&
            get_constr_viol_tol(solver.ncl, r) <= options.constr_viol_tol
        )
            ncl_status = MadNLP.SOLVE_SUCCEEDED
            break
        # Check infeasibility
        elseif (ncl.ρk[] >= options.rho_max) && (pr_feas > options.opt_tol)
            ncl_status = MadNLP.INFEASIBLE_PROBLEM_DETECTED
            break
        end
        iter += 1
    end

    # Loop exhausted without convergence
    if ncl_status == MadNLP.INITIAL
        ncl_status = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    end

    ipm.cnt.total_time = time() - start_time

    return NCLStats(solver, ncl_status)
end

function madncl(
    nlp::NLPModels.AbstractNLPModel;
    options...
)
    solver = NCLSolver(nlp; options...)
    return solve!(solver)
end

