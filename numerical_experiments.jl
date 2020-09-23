using CSV, DataFrames
using JuMP, Ipopt
using LinearAlgebra
using SparseArrays
using Mosek, MosekTools
using Distributions
using StatsBase


include("main.jl")

# experiment settings
settings = Dict(:ψ_𝛑 => 0, :ψ_φ => 0, :ε => 0.01, :σ => 0.1, :det => false)
# set network case
case = "case_48"
# extarct network data
gas_data        = load_data(case)
# solve non-convex gas network optimization
sol_non_convex  = gas_non_convex(gas_data)
# obtain linearization data
lin_res         = linearization(gas_data,sol_non_convex[:model])
# extract forecast data
forecast        = forecast_data(gas_data,settings)
# solve chance constrained gas network optimization
sol_stochastic  = gas_cc(gas_data,lin_res,forecast,settings)
# run out of sample analysis
sol_ofs         = out_of_sample(gas_data,forecast,sol_stochastic)
# get dual solution to the chance constrained gas network optimization
sol_dual        = stochastic_dual_solution(gas_data,sol_stochastic,lin_res,forecast,settings)

# base cc solution
exp_cost_base = sol_stochastic[:cost]
sum_s_ρ_base = sum([var(sol_ofs[:ρ][n,:]) for n in gas_data[:N]])
sum_s_φ_base = sum([var(sol_ofs[:φ][l,:]) for l in gas_data[:E]])
sum_κ_½_valv_base = sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x<0, gas_data[:κ̲])])
sum_κ_½_comp_base = sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x>0, gas_data[:κ̅])])

# compute cost-pressure_var trade-offs
settings = Dict(:ψ_𝛑 => 0, :ψ_φ => 0, :ε => 0.01, :σ => 0.1, :det => false)
pres_cost_var_trade_offs = DataFrame(ψ_𝛑=Any[],cost=Any[],var_π=Any[],var_φ=Any[],inf=Any[],κ_comp=Any[],κ_valv=Any[],Δϑ_mean=Any[],Δκ_mean=Any[])
ψ_π = [0.001 0.01 0.1]
for i in ψ_π
    settings[:ψ_𝛑] = i
    sol_stochastic = gas_cc(gas_data,lin_res,forecast,settings)
    sol_ofs        = out_of_sample(gas_data,forecast,sol_stochastic)
    sol_proj       = projection_analysis(gas_data,forecast,sol_ofs)

    exp_cost = sol_stochastic[:cost] / exp_cost_base * 100
    sum_s_ρ = sum([var(sol_ofs[:ρ][n,:]) for n in gas_data[:N]]) / sum_s_ρ_base * 100
    sum_s_φ = sum([var(sol_ofs[:φ][l,:]) for l in gas_data[:E]]) / sum_s_φ_base * 100
    inf_per = sol_ofs[:ε_stat]*100
    sum_κ_½_valv = round(sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x<0, gas_data[:κ̲])]))
    sum_κ_½_comp = round(sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x>0, gas_data[:κ̅])]))
    Δϑ_mean = sol_proj[1]
    Δκ_mean = sol_proj[2]

    push!(pres_cost_var_trade_offs,[i,exp_cost,sum_s_ρ,sum_s_φ,inf_per,sum_κ_½_comp,sum_κ_½_valv,Δϑ_mean,Δκ_mean])
end

# compute cost-flow_var trade-offs
settings = Dict(:ψ_𝛑 => 0, :ψ_φ => 0, :ε => 0.01, :σ => 0.1, :det => false)
flow_cost_var_trade_offs = DataFrame(ψ_φ=Any[],cost=Any[],var_π=Any[],var_φ=Any[],inf=Any[],κ_comp=Any[],κ_valv=Any[],Δϑ_mean=Any[],Δκ_mean=Any[])
ψ_φ = [1 10 100]
for i in ψ_φ
    settings[:ψ_φ] = i
    sol_stochastic = gas_cc(gas_data,lin_res,forecast,settings)
    sol_ofs        = out_of_sample(gas_data,forecast,sol_stochastic)
    sol_proj       = projection_analysis(gas_data,forecast,sol_ofs)

    exp_cost = sol_stochastic[:cost]/exp_cost_base*100
    sum_s_ρ = sum([var(sol_ofs[:ρ][n,:]) for n in gas_data[:N]])/sum_s_ρ_base*100
    sum_s_φ = sum([var(sol_ofs[:φ][l,:]) for l in gas_data[:E]])/sum_s_φ_base*100
    inf_per = sol_ofs[:ε_stat]*100
    sum_κ_½_valv = round(sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x<0, gas_data[:κ̲])]))
    sum_κ_½_comp = round(sum([sqrt(abs(sol_stochastic[:κ][l])) for l in findall(x->x>0, gas_data[:κ̅])]))
    Δϑ_mean = sol_proj[1]
    Δκ_mean = sol_proj[2]

    push!(flow_cost_var_trade_offs,[i,exp_cost,sum_s_ρ,sum_s_φ,inf_per,sum_κ_½_comp,sum_κ_½_valv,Δϑ_mean,Δκ_mean])
end

@info("Pressure variance-aware results:")
@show pres_cost_var_trade_offs

@info("Flow variance-aware results:")
@show flow_cost_var_trade_offs


# compute revenues
# deterministic
settings = Dict(:ψ_𝛑 => 0, :ψ_φ => 0, :ε => 0.01, :σ => 0.1, :det => true)
sol_stochastic  = gas_cc(gas_data,lin_res,forecast,settings)
sol_dual        = stochastic_dual_solution(gas_data,sol_stochastic,lin_res,forecast,settings)
@info("Deterministic policies:")
@show sol_dual[:Revenue_decomposition]

# variance-agnostic
settings = Dict(:ψ_𝛑 => 0, :ψ_φ => 0, :ε => 0.01, :σ => 0.1, :det => false)
sol_stochastic  = gas_cc(gas_data,lin_res,forecast,settings)
sol_dual        = stochastic_dual_solution(gas_data,sol_stochastic,lin_res,forecast,settings)
@info("Variance-agnostic policies:")
@show sol_dual[:Revenue_decomposition]

# variance-aware
settings = Dict(:ψ_𝛑 => 0.1, :ψ_φ => 100, :ε => 0.01, :σ => 0.1, :det => false)
sol_stochastic  = gas_cc(gas_data,lin_res,forecast,settings)
sol_dual        = stochastic_dual_solution(gas_data,sol_stochastic,lin_res,forecast,settings)
@info("Variance-aware policies:")
@show sol_dual[:Revenue_decomposition]
