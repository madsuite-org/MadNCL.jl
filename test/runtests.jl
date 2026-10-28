
using Test
using LinearAlgebra
using MadNLP
using MadNCL
using NLPModels, NLPModelsJuMP
using FiniteDiff
using MadNLPTests
using JuMP

using CUDA

include("hs15.jl")
include("dummy_qp.jl")

function test_nlp_model(ncl)
    n, m = NLPModels.get_nvar(ncl), NLPModels.get_ncon(ncl)

    x0 = NLPModels.get_x0(ncl)
    g = similar(x0, n)
    c = similar(x0, m)

    # Objective
    @test NLPModels.obj(ncl, x0) isa Number
    # Constraints
    NLPModels.cons!(ncl, x0, c)

    # Gradient
    NLPModels.grad!(ncl, x0, g)

    # Test with FiniteDiff
    fun_grad_fd = x -> NLPModels.obj(ncl, x)
    g_fd = FiniteDiff.finite_difference_gradient(fun_grad_fd, x0)
    @test g ≈ g_fd

    # Jacobian
    nnzj = NLPModels.get_nnzj(ncl)
    jac_i = zeros(Int, nnzj)
    jac_j = zeros(Int, nnzj)
    jac_v = zeros(Float64, nnzj)
    NLPModels.jac_structure!(ncl, jac_i, jac_j)
    NLPModels.jac_coord!(ncl, x0, jac_v)
    # Extract Jacobian
    J = zeros(Float64, m, n)
    for (i, j, v) in zip(jac_i, jac_j, jac_v)
        J[i, j] += v
    end

    # Test with FiniteDiff
    fun_jac_fd = x -> NLPModels.cons(ncl, x)
    J_fd = FiniteDiff.finite_difference_jacobian(fun_jac_fd, x0)
    @test J ≈ J_fd atol=1e-5

    # Jtprod
    v = rand(m)
    jtv = zeros(n)
    NLPModels.jtprod!(ncl, x0, v, jtv)
    @test jtv ≈ J' * v atol=1e-5
    # Jprod
    v = rand(n)
    jv = zeros(m)
    NLPModels.jprod!(ncl, x0, v, jv)
    @test jv == J * v

    # Hessian
    y = rand(m)
    nnzh = NLPModels.get_nnzh(ncl)
    hess_i = zeros(Int, nnzh)
    hess_j = zeros(Int, nnzh)
    hess_v = zeros(Float64, nnzh)
    NLPModels.hess_structure!(ncl, hess_i, hess_j)
    NLPModels.hess_coord!(ncl, x0, y, hess_v)
    # Unpack
    H = zeros(Float64, n, n)
    for (i, j, v) in zip(hess_i, hess_j, hess_v)
        H[i, j] += v
        if i != j
            H[j, i] += v
        end
    end
    # Allocate temporary buffers to avoid NaN in FiniteDiff
    g_tmp = similar(x0)
    j_tmp = similar(x0)
    function fun_hess_fd(x)
        fill!(g_tmp, 0.0)
        NLPModels.grad!(ncl, x, g_tmp)
        NLPModels.jtprod!(ncl, x, y, j_tmp)
        return g_tmp .+ j_tmp
    end
    H_fd = FiniteDiff.finite_difference_jacobian(fun_hess_fd, x0)
    @test Symmetric(H, :L) ≈ H_fd rtol=1e-6
    return
end

@testset "Test regularized models" begin
    nlp = HS15Model()
    n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
    @testset "NCLModel" begin
        ncl = MadNCL.NCLModel(nlp)
        @test NLPModels.get_nvar(ncl) == n + m
        @test NLPModels.get_ncon(ncl) == m
        ncl.ρk[] = 10.0
        test_nlp_model(ncl)
    end
    @testset "ScaledModel" begin
        snlp = MadNCL.ScaledModel(nlp)
        test_nlp_model(snlp)
    end
end

@testset "Test NCL algorithm" begin
    @testset "HS15" begin
        nlp = HS15Model()
        # Test we recover original solution for ρ ≫ 1
        n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
        res_original = MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
        ncl = MadNCL.NCLModel(nlp)
        ncl.yk .= 0
        ncl.ρk[] = 1e9
        res_regularized = MadNLP.madnlp(ncl; print_level=MadNLP.ERROR)
        @test res_regularized.solution[1:n] ≈ res_original.solution atol=1e-6

        # Test full NCL algorithm.
        stats = MadNCL.madncl(nlp; print_level=MadNLP.ERROR)
        @test stats.status == MadNLP.SOLVE_SUCCEEDED

        # Test full NCL algorithm (without scaling).
        ncl_opt = MadNCL.NCLOptions{Float64}(; scaling=false)
        stats = MadNCL.madncl(nlp; ncl_options=ncl_opt, print_level=MadNLP.ERROR)
        @test stats.status == MadNLP.SOLVE_SUCCEEDED
    end
end

@testset "KKT system reformulation" begin
    nlp = HS15Model()
    ncl = MadNCL.NCLModel(nlp)
    res_original = MadNLP.madnlp(ncl; print_level=MadNLP.ERROR, linear_solver=LapackCPUSolver)
    @testset "[KKT] $(KKT)" for KKT in [
        MadNCL.K2rAuglagKKTSystem,
        MadNCL.K1sAuglagKKTSystem,
    ]
        # Test KKT formulation
        linear_solver = MadNLP.LapackCPUSolver
        cb = MadNLP.create_callback(
            MadNLP.SparseCallback, ncl,
        )
        kkt = MadNLP.create_kkt_system(
            KKT,
            cb,
            linear_solver;
        )
        MadNLPTests.test_kkt_system(kkt, cb)

        # Test optimization runs to completion
        solver = MadNLP.MadNLPSolver(
            ncl;
            linear_solver=LapackCPUSolver,
            kkt_system=KKT,
            print_level=MadNLP.ERROR,
        )
        stats = MadNLP.solve!(solver)
        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.iter == res_original.iter
        @test stats.objective ≈ res_original.objective
    end
end

@testset "Precision $(T) ($(kkt))" for T in (Float32, Float64), kkt in [
    MadNCL.K2rAuglagKKTSystem,
    MadNCL.K1sAuglagKKTSystem,
]
    nlp = MadNLPTests.HS15Model(; T=T, x0=T[1, 1])
    tol = T == Float32 ? T(1e-3) : T(1e-6)
    solver = MadNCL.NCLSolver(
        nlp;
        ncl_options=MadNCL.NCLOptions{T}(verbose=false, opt_tol=tol, feas_tol=tol),
        print_level=MadNLP.ERROR,
        kkt_system=kkt,
        linear_solver=LDLSolver,
    )
    stats = @inferred MadNCL.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test stats.objective isa T
    @test eltype(stats.solution) == T
    @test eltype(stats.multipliers) == T
    @test stats.solution ≈ T[0.5, 2] rtol=sqrt(tol)
end

@testset "Fixed variables ($(kkt))" for kkt in [
    MadNCL.K2rAuglagKKTSystem,
    MadNCL.K1sAuglagKKTSystem,
]
    nlp = HS15Model()
    # Set LB = UB to fix the first variable
    nlp.meta.lvar[1] = 0.5
    ref_results = MadNLP.madnlp(nlp; print_level=MadNLP.ERROR)
    stats = MadNCL.madncl(nlp; print_level=MadNLP.ERROR, kkt_system=kkt)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test stats.objective ≈ ref_results.objective rtol=1e-6
    @test stats.multipliers ≈ ref_results.multipliers rtol=1e-6
    @test stats.solution ≈ ref_results.solution rtol=1e-6
    @test stats.multipliers_L ≈ ref_results.multipliers_L rtol=1e-6
    @test stats.multipliers_U ≈ ref_results.multipliers_U rtol=1e-6
end

# Test taken from MadNLPTests
@testset "Max sense optimization problem" begin
    model = Model()
    @variable(model, 0.0 <= x[1:3])
    @constraint(model, x[1] + x[2] + x[3] == 1.0)
    @objective(model, Max, x[1] + 2*x[2] + 3*x[3])

    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    results = MadNCL.madncl(nlp; print_level=MadNLP.ERROR)

    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test results.objective ≈ 3.0 rtol=1e-6
    @test results.solution ≈ [0.0, 0.0, 1.0] rtol=1e-6
    @test results.multipliers[1] ≈ -3.0 rtol=1e-6
    @test results.multipliers_L[1] ≈ -2.0 rtol=1e-6
    @test results.multipliers_L[2] ≈ -1.0 rtol=1e-6
    @test results.multipliers_L[3] ≈ 0.0 atol=1e-6
end

if CUDA.has_cuda()
    include("cuda_wrapper.jl")
end

include("termination.jl")
