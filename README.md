# MadNCL.jl

[![Run tests](https://github.com/madsuite-org/MadNCL.jl/actions/workflows/action.yml/badge.svg)](https://github.com/madsuite-org/MadNCL.jl/actions/workflows/action.yml)

An implementation of [Algorithm NCL](https://link.springer.com/chapter/10.1007/978-3-319-90026-1_8) in Julia.

MadNCL is built as [MadNLP](https://github.com/madsuite-org/MadNLP.jl)'s extension, and supports the solution of
large-scale nonlinear programs on GPUs. MadNCL is particularly good at solving infeasible
or degenerate optimization problems.

## Installation

MadNCL can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add MadNCL
pkg> test MadNCL
```

## Quickstart

MadNCL leverages [JuliaSmoothOptimizers](https://jso.dev)'s ecosystem.
For instance, you can solve any instance in [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl/) simply as:

```julia
using MadNCL
using CUTEst
nlp = CUTEstModel("HS15")
results = madncl(nlp)
```

You gain more fine-grained control on the solver by tuning the options:

```julia
# Specify NCL's options
ncl_options = MadNCL.NCLOptions{Float64}(
    verbose=true,       # print convergence logs
    scaling=false,      # specify if we should scale the problem
    opt_tol=1e-8,       # tolerance on dual infeasibility
    feas_tol=1e-8,      # tolerance on primal infeasibility
    rho_init=1e1,       # initial augmented Lagrangian penalty
    max_auglag_iter=20, # maximum number of outer iterations
)
# MadNLP's options are passed directly to madncl:
results = madncl(
    nlp;
    ncl_options=ncl_options,
    linear_solver=LDLSolver,   # factorize the KKT system with a LDL decomposition
    print_level=MadNLP.INFO,   # activate logs inside MadNLP
)
```

## JuMP

MadNCL does not yet provide an extension for [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl), but a JuMP model can be wrapped
as a `MathOptNLPModel` using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl).

```julia
using JuMP
using NLPModelsJuMP
using MadNCL

jm = Model()
x0 = [10, 10, 10, 10]
@variable(jm, x[i = 1:4], start = x0[i])
@constraint(jm, x[1]^2 - x[2] - x[4]^2 == 0)
@constraint(jm, x[2] - x[1]^3 - x[3]^2 == 0)
@objective(jm, Min, -x[1])

nlp = MathOptNLPModel(jm)
results = madncl(nlp)
```

## GPU support

MadNCL supports natively the solution of nonlinear programs on the GPU using MadNLPGPU.
To evaluate your model on the GPU, we recommend using [ExaModels](https://github.com/madsuite-org/ExaModels.jl).
For instance, you can implement the instance `elec` from the [COPS benchmark](https://www.mcs.anl.gov/~more/cops/) directly as:

```julia
using ExaModels

function elec_model(np; seed = 2713, T = Float64, backend = nothing, kwargs...)
    Random.seed!(seed)
    # Set the starting point to a quasi-uniform distribution of electrons on a unit sphere
    theta = (2pi) .* rand(np)
    phi = pi .* rand(np)

    core = ExaModels.ExaCore(T; backend= backend)
    x = ExaModels.variable(core, 1:np; start = [cos(theta[i])*sin(phi[i]) for i=1:np])
    y = ExaModels.variable(core, 1:np; start = [sin(theta[i])*sin(phi[i]) for i=1:np])
    z = ExaModels.variable(core, 1:np; start = [cos(phi[i]) for i=1:np])
    # Coulomb potential
    itr = [(i,j) for i in 1:np-1 for j in i+1:np]
    ExaModels.objective(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in itr)
    # Unit-ball
    ExaModels.constraint(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i=1:np)

    return ExaModels.ExaModel(core; kwargs...)
end
```

You can instantiate the model and solve it on the GPU using CUDA and MadNLPGPU, respectively:
```julia
using CUDA
using MadNLPGPU

nlp = elec_model(100; backend=CUDABackend())

results = madncl(
    nlp;
    print_level=MadNLP.INFO,                # activate logs inside MadNLP
    kkt_system=MadNCL.K2rAuglagKKTSystem,   # we need to reformulate the Newton system inside MadNCL
    linear_solver=MadNLPGPU.CUDSSSolver,    # factorize the KKT system on the GPU using NVIDIA cuDSS
)
```

## Citing MadNCL.jl

If you use MadNCL.jl in your research, we would greatly appreciate your citing it.

```bibtex
@article{MadNCL,
  title   = {{MadNCL: A GPU Implementation of Algorithm NCL for Large-Scale, Degenerate Nonlinear Programs}},
  author  = {Alexis Montoison and François Pacaud and Michael Saunders and Sungho Shin and Dominique Orban},
  journal = {arXiv preprint arXiv:2510.05885},
  year    = {2025},
}
```
