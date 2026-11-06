module MadNCLCUDAExt

using LinearAlgebra
using SparseArrays
import MadCoreGPU
import CUDA: cuSPARSE, CUDACore
import CUDA: CuVector, CUDABackend
import KernelAbstractions: synchronize
import MadNLPCore
import MadNCL

function MadNLPCore.transfer!(
    dest::cuSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLPCore.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    fill!(nonzeros(dest), zero(Tv))
    if length(map) > 0
        MadNCL._transfer_to_map!(CUDABackend())(nonzeros(dest), map, src.V; ndrange=length(map))
        synchronize(CUDABackend())
    end
    return
end

function MadNCL.symul!(
    y::CuVector{T},
    A::cuSPARSE.CuSparseMatrixCSC{T},
    x::CuVector{T},
    alpha::Number,
    beta::Number,
) where {T}
    m, n = size(A)
    mul!(y, A , x, alpha, beta)
    mul!(y, A', x, alpha, one(T))
    MadNCL._remove_diagonal_kernel!(CUDABackend())(y, A.colPtr, A.rowVal, A.nzVal, x, alpha; ndrange=n)
    synchronize(CUDABackend())
    return y
end

function MadNLPCore.build_condensed_aug_coord!(
    kkt::MadNCL.K1sAuglagKKTSystem{T,VT,MT},
) where {T,VT,MT<:cuSPARSE.CuSparseMatrixCSC{T}}
    fill!(kkt.aug_com.nzVal, zero(T))
    if length(kkt.hptr) > 0
        MadCoreGPU._transfer_hessian_kernel!(CUDABackend())(
            kkt.aug_com.nzVal,
            kkt.hptr,
            kkt.hess_com.nzVal;
            ndrange = length(kkt.hptr),
        )
        synchronize(CUDABackend())
    end
    if length(kkt.dptr) > 0
        MadCoreGPU._transfer_hessian_kernel!(CUDABackend())(
            kkt.aug_com.nzVal,
            kkt.dptr,
            kkt.pr_diag_reduced;
            ndrange = length(kkt.dptr),
        )
        synchronize(CUDABackend())
    end
    if length(kkt.ext.jptrptr) > 1 # otherwise error is thrown
        MadCoreGPU._transfer_jtsj_kernel!(CUDABackend())(
            kkt.aug_com.nzVal,
            kkt.jptr,
            kkt.ext.jptrptr,
            kkt.jt_csc.nzVal,
            kkt.diag_buffer;
            ndrange = length(kkt.ext.jptrptr) - 1,
        )
        synchronize(CUDABackend())
    end
    return
end

end

