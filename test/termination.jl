# termination = :madnlp: the returned point satisfies MadNLP's own test for the original problem
using MadNLPTests

function madnlp_residuals_at(nlp, stats; tol = 1e-8)
    term = MadNCL.MadNLPTermination(nlp; tol = tol)
    sgn = NLPModels.get_minimize(nlp) ? 1.0 : -1.0
    # stats.multipliers follow the NLPModels convention for f (not for the objective MadNLP
    # minimizes), so a maximization flips them back
    y = sgn .* stats.multipliers; zl = sgn .* stats.multipliers_L; zu = sgn .* stats.multipliers_U
    return MadNCL.madnlp_residuals(term, nlp, stats.solution, y, zl, zu)
end

@testset "Termination on MadNLP's criterion" begin
    tol = 1e-8
    for (name, nlp) in [("HS15", HS15Model()), ("DummyQP", DenseDummyQP(zeros(100); m = 10))]
        # MadNLP's own solution passes the test as implemented here
        ref = MadNLP.madnlp(nlp; tol = tol, print_level = MadNLP.ERROR)
        @test ref.status == MadNLP.SOLVE_SUCCEEDED
        pr, du, cc = madnlp_residuals_at(nlp, ref; tol = tol)
        @test max(pr, du, cc) <= 10 * tol
        # madncl with the MadNLP criterion returns a point that passes it
        opts = MadNCL.NCLOptions{Float64}(; verbose = false, termination = :madnlp, tol = tol)
        res = MadNCL.madncl(nlp; ncl_options = opts, tol = tol, print_level = MadNLP.ERROR)
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        pr, du, cc = madnlp_residuals_at(nlp, res; tol = tol)
        @test max(pr, du, cc) <= tol
    end
    @test_throws ArgumentError MadNCL.NCLSolver(HS15Model(); ncl_options = MadNCL.NCLOptions{Float64}(termination = :other))
end
