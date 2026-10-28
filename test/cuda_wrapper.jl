
using CUDA
using CUDSS
using MadNLPGPU

function test_model_gpu(ncl_cpu, ncl_gpu)
    n, m = NLPModels.get_nvar(ncl_gpu), NLPModels.get_ncon(ncl_gpu)
    @test n == NLPModels.get_nvar(ncl_cpu)
    @test m == NLPModels.get_ncon(ncl_cpu)
    # initial position
    x0_cpu = NLPModels.get_x0(ncl_cpu)
    x0_gpu = NLPModels.get_x0(ncl_gpu)
    @test x0_cpu ≈ Array(x0_gpu)
    # objective
    val_cpu = NLPModels.obj(ncl_cpu, x0_cpu)
    val_gpu = NLPModels.obj(ncl_gpu, x0_gpu)
    @test val_cpu ≈ val_gpu
    # constraint
    c_cpu = NLPModels.cons(ncl_cpu, x0_cpu)
    c_gpu = NLPModels.cons(ncl_gpu, x0_gpu)
    @test c_cpu ≈ Array(c_gpu)
    # gradient
    g_cpu = NLPModels.grad(ncl_cpu, x0_cpu)
    g_gpu = NLPModels.grad(ncl_gpu, x0_gpu)
    @test g_cpu ≈ Array(g_gpu)
    # Jacobian sparsity pattern
    nnzj_cpu = NLPModels.get_nnzj(ncl_cpu)
    nnzj_gpu = NLPModels.get_nnzj(ncl_gpu)
    @test nnzj_cpu == nnzj_gpu
    i_cpu = zeros(Int, nnzj_cpu)
    j_cpu = zeros(Int, nnzj_cpu)
    NLPModels.jac_structure!(ncl_cpu, i_cpu, j_cpu)
    i_gpu = CUDA.zeros(Int, nnzj_gpu)
    j_gpu = CUDA.zeros(Int, nnzj_gpu)
    NLPModels.jac_structure!(ncl_gpu, i_gpu, j_gpu)
    @test i_cpu == Array(i_gpu)
    @test j_cpu == Array(j_gpu)
    # Jacobian values
    jac_cpu = zeros(Float64, nnzj_cpu)
    NLPModels.jac_coord!(ncl_cpu, x0_cpu, jac_cpu)
    jac_gpu = CUDA.zeros(Float64, nnzj_gpu)
    NLPModels.jac_coord!(ncl_gpu, x0_gpu, jac_gpu)
    @test jac_cpu ≈ Array(jac_gpu)
    # Hessian sparsity pattern
    nnzh_cpu = NLPModels.get_nnzh(ncl_cpu)
    nnzh_gpu = NLPModels.get_nnzh(ncl_gpu)
    @test nnzh_cpu == nnzh_gpu
    i_cpu = zeros(Int, nnzh_cpu)
    j_cpu = zeros(Int, nnzh_cpu)
    NLPModels.hess_structure!(ncl_cpu, i_cpu, j_cpu)
    i_gpu = CUDA.zeros(Int, nnzh_gpu)
    j_gpu = CUDA.zeros(Int, nnzh_gpu)
    NLPModels.hess_structure!(ncl_gpu, i_gpu, j_gpu)
    @test i_cpu == Array(i_gpu)
    @test j_cpu == Array(j_gpu)
    # Hessian values
    hess_gpu = CUDA.zeros(Float64, nnzh_gpu)
    y_gpu = CUDA.ones(Float64, m)
    NLPModels.hess_coord!(ncl_gpu, x0_gpu, y_gpu, hess_gpu)
    hess_cpu = zeros(Float64, nnzh_cpu)
    y_cpu = ones(Float64, m)
    NLPModels.hess_coord!(ncl_cpu, x0_cpu, y_cpu, hess_cpu)
    @test hess_cpu ≈ Array(hess_gpu)
    return
end

@testset "[CUDA] Test NCL model" begin
    # N.B.: n should not be too small to avoid fatal bug with cuDSS
    #       (solver hanging as the matrix is too small).
    n, m = 100, 5
    # Host evaluator
    nlp_cpu = DenseDummyQP(zeros(Float64, n); m=m)
    ncl_cpu = MadNCL.NCLModel(nlp_cpu)
    # Device evaluator
    nlp_gpu = DenseDummyQP(CUDA.zeros(Float64, n); m=m)
    ncl_gpu = MadNCL.NCLModel(nlp_gpu)

    # Test instantiation of NCLModel on the GPU.
    test_model_gpu(ncl_cpu, ncl_gpu)

    # Test NCL algorithm on the GPU.
    for KKTSystem in [
        MadNCL.K2rAuglagKKTSystem,
        MadNCL.K1sAuglagKKTSystem,
    ]
        res = MadNCL.madncl(
            nlp_gpu;
            print_level=MadNLP.ERROR,
            kkt_system=KKTSystem,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
        )
        @test res.status == MadNLP.SOLVE_SUCCEEDED
    end

    # Termination on MadNLP's criterion, evaluated on the device
    tol = 1e-8
    opts = MadNCL.NCLOptions{Float64}(; verbose = false, termination = :madnlp, tol = tol)
    res = MadNCL.madncl(
        nlp_gpu;
        ncl_options=opts,
        tol=tol,
        print_level=MadNLP.ERROR,
        kkt_system=MadNCL.K2rAuglagKKTSystem,
        linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
    )
    @test res.status == MadNLP.SOLVE_SUCCEEDED
    term = MadNCL.MadNLPTermination(nlp_gpu; tol = tol)
    pr, du, cc = MadNCL.madnlp_residuals(term, nlp_gpu, res.solution, res.multipliers, res.multipliers_L, res.multipliers_U)
    @test max(pr, du, cc) <= tol
    ref = MadNLP.madnlp(nlp_cpu; tol = tol, print_level = MadNLP.ERROR)
    @test isapprox(res.objective, ref.objective; rtol = 1e-6)
end

