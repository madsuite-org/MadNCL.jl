module MadNCL

import LinearAlgebra: norm
import LinearAlgebra: dot, axpy!, mul!
import LinearAlgebra: Symmetric
import SparseArrays: sparsevec, nonzeros
import SparseArrays: SparseMatrixCSC
import Printf: @printf
import NLPModels
import MadNLP
import MadNLP: AbstractExecutionStats, getStatus

# CUDA related deps
import Atomix
import KernelAbstractions: @kernel, @index

export madncl

include("utils.jl")
include("Models/ncl.jl")
include("Models/scaled.jl")

include("KKT/k1s.jl")
include("KKT/k2r.jl")
include("KKT/scaled_k2r.jl")

include("solver.jl")

MadNLP.madsuite(::Val{:madncl}, args...; kwargs...) = madncl(args...; kwargs...)

end
