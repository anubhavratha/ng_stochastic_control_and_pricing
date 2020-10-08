#!/usr/bin/env julia
using CSV, DataFrames
using JuMP, Ipopt
using LinearAlgebra
using SparseArrays
using Mosek, MosekTools
using Distributions
using StatsBase

### load functions
# data scripts
include("scr/fun_data.jl")
include("scr/fun_auxiliary.jl")
# optimization models + supporting functions
include("scr/fun_opt_models.jl")
include("scr/fun_linearization.jl")
# out-of-sample + projection functions
include("scr/fun_post_processing.jl")

### experiment settings
exp_settings = Dict(
# test case
:case => "case_48",
# pressure variance penalty
:ψ_𝛑 => 0.1,
# flow variance penalty
:ψ_φ => 0,
# prescribed constraint violation probability
:ε => 0.01,
# gas extraction standard deviation (% of the nominal level)
:σ => 0.1,
# det: true - optimize determenistic policies, false - optimize chance-constrained policies
:det => false,
# comp: true - compressor recourse on, false - compressor recourse off
:comp => true,
# valv: true - valve recourse on, false - valve recourse off
:valv => true,
# proj: true - run projection analysis, false - don't run
:proj => false,
# sample compexity
:S => 1000
)

### display input
println()
println("------ Experiment settings ------")
exp_settings[:det] == true ?
println("Policy optimization      :     deterministic") :
println("Policy optimization      :     chance-constrained")
println("Variance penalty         :     ψ_𝛑 = $(exp_settings[:ψ_𝛑]) ... ψ_φ = $(exp_settings[:ψ_φ])")
println("Violation probability    :     ε = $(exp_settings[:ε])")
println("Standard deviation       :     σ = $(exp_settings[:σ])")
println("Active pipeline recourse :     comp = $(exp_settings[:comp]) ... valve = $(exp_settings[:valv])")
println("Out of sample setting    :     proj = $(exp_settings[:proj]) ... S = $(exp_settings[:S])")

### run experiment
println()
println("------ Run experiment ------")
# extarct network data
net_data        = load_network_data(exp_settings[:case])
# solve non-convex gas network optimization
sol_non_convex  = non_convex_opt(net_data)
# obtain linearization data
lin_res         = linearization(net_data,sol_non_convex[:model])
# extract forecast data
forecast        = extract_forecast_data(net_data,exp_settings)
# solve chance-constrained gas network optimization
sol_stochastic  = chance_con_opt(net_data,lin_res,forecast,exp_settings)
# get dual solution to the chance-constrained gas network optimization
sol_dual        = stochastic_dual_solution(net_data,sol_stochastic,lin_res,forecast,exp_settings)
# run out of sample analysis
sol_ofs         = out_of_sample(net_data,forecast,sol_stochastic)
# run projection analysis
exp_settings[:proj] == true ? sol_proj        = projection_analysis(net_data,forecast,sol_ofs) : NaN

### display output
println()
println("------ Experiment results ------")
println("Expected cost (\$)              :   $(round(sol_stochastic[:cost],digits=1))")
println("Total pressure variance (MPa²) :   $(round(sum([var(sol_ofs[:ρ][n,:]) for n in net_data[:N]]),digits=1))")
println("Total flow variance (BMSCFD²)  :   $(round(sum([var(sol_ofs[:φ][l,:]) for l in net_data[:E]]),digits=1))")
println("Average comp deployment (kPa)  :   $(round(sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x>0, net_data[:κ̅])])))")
println("Average valv deployment (kPa)  :   $(round(sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x<0, net_data[:κ̲])])))")
println("Emp. constraint violation (%)  :   $(round(sol_ofs[:ε_stat]*100,digits=5))")
exp_settings[:proj] == true ?
println("Average injection proj (MMSCFD) :   $(round(sol_proj[:Δϑ_mean],digits=3))") : NaN
exp_settings[:proj] == true ?
println("Average regulation proj (kPa)   :   $(round(sol_proj[:Δκ_mean],digits=3))") : NaN
println("Total revenue of suppliers (\$) :   $(round(sol_dual[:R_inj],digits=3))")
println("Total revenue of act. pipes (\$):   $(round(sol_dual[:R_act],digits=3))")
println("Total revenue of consumers (\$) :   $(round(sol_dual[:R_con],digits=3))")
