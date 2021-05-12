function out_of_sample(net_data,forecast,sol_stochastic)
    """
    runs out-of-sample analysis
    """
    φ̃ = zeros(length(net_data[:E]),forecast[:S])
    ϑ̃ = zeros(length(net_data[:N]),forecast[:S])
    κ̃ = zeros(length(net_data[:E]),forecast[:S])
    π̃ = zeros(length(net_data[:N]),forecast[:S])
    ρ̃ = zeros(length(net_data[:N]),forecast[:S])
    exp_cost = zeros(forecast[:S])

    for s in 1:forecast[:S]
        φ̃[:,s] = sol_stochastic[:φ] .+ (lin_res[:ς̀2]*(sol_stochastic[:α] .- diagm(ones(length(net_data[:N])))) .- lin_res[:ς̀3]*sol_stochastic[:β])*forecast[:ξ][:,s]
        ϑ̃[:,s] = sol_stochastic[:ϑ] .+ sol_stochastic[:α]*forecast[:ξ][:,s]
        κ̃[:,s] = sol_stochastic[:κ] .+ sol_stochastic[:β]*forecast[:ξ][:,s]
        π̃[:,s] = sol_stochastic[:𝛑] .+ (lin_res[:ς̆2]*(sol_stochastic[:α] .- lin_res[:ς̂3]*sol_stochastic[:β] .- diagm(ones(length(net_data[:N])))))*forecast[:ξ][:,s]
        ρ̃[:,s] = sqrt.(max.(π̃[:,s],0))
        exp_cost[s] = ϑ̃[:,s]'*diagm(vec(net_data[:c]))*ϑ̃[:,s]
    end

    inf_flag = zeros(forecast[:S])
    num_tolerance = 0.001
    for s in 1:forecast[:S]
        for n in net_data[:N]
            if n in findall(x->x>0, net_data[:ϑ̅])
                ϑ̃[n,s] >= net_data[:ϑ̅][n] + num_tolerance ? inf_flag[s] = 1 : NaN
                ϑ̃[n,s] <= 0 - num_tolerance ? inf_flag[s] = 1 : NaN
            end
            ρ̃[n,s] >= net_data[:ρ̅][n] + num_tolerance ? inf_flag[s] = 1 : NaN
            ρ̃[n,s] <= net_data[:ρ̲][n] - num_tolerance ? inf_flag[s] = 1 : NaN
        end
        for p in net_data[:E_a]
            φ̃[p,s] <= 0  - num_tolerance ? inf_flag[s] = 1 : NaN
            κ̃[p,s] >= net_data[:κ̅][p]  + num_tolerance ? inf_flag[s] = 1 : NaN
            κ̃[p,s] <= net_data[:κ̲][p]  - num_tolerance ? inf_flag[s] = 1 : NaN
        end
    end
    @info("done out-of-sample results")
    return Dict(:φ => φ̃, :ϑ => ϑ̃, :κ => κ̃, :𝛑 => π̃, :ρ => ρ̃,
                :cost => mean(exp_cost), :ε_stat => sum(inf_flag)/forecast[:S])
end
function projection_opt(net_data,forecast,sol_ofs,s)
    """
    solves projection problem for a single uncertainty sample
    """
    N_ϑ = findall(x->x>0, net_data[:ϑ̅])
    N_κ = net_data[:E_a]
    # build model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(net_data[:N])])
    @variable(model, φ[1:length(net_data[:E])])
    @variable(model, κ[1:length(net_data[:E])])
    @variable(model, ϑ[1:length(net_data[:N])])
    # minimize gas production cost
    @objective(model, Min, (sol_ofs[:ϑ][N_ϑ,s] .- ϑ[N_ϑ])' * diagm(ones(length(N_ϑ))) * (sol_ofs[:ϑ][N_ϑ,s] .- ϑ[N_ϑ])
                            + (sol_ofs[:κ][N_κ,s] .- κ[N_κ])' * diagm(ones(length(N_κ))) * (sol_ofs[:κ][N_κ,s] .- κ[N_κ])
                )
    # gas variable limits
    @constraint(model, inj_lim_max[i=net_data[:N]], ϑ[i] <= net_data[:ϑ̅][i])
    @constraint(model, inj_lim_min[i=net_data[:N]], ϑ[i] >= net_data[:ϑ̲][i])
    @constraint(model, pre_lim_max[i=net_data[:N]], 𝛑[i] <= net_data[:ρ̅][i]^2)
    @constraint(model, pre_lim_min[i=net_data[:N]], 𝛑[i] >= net_data[:ρ̲][i]^2)
    @constraint(model, com_lim_max[i=net_data[:E]], κ[i] <= net_data[:κ̅][i])
    @constraint(model, com_lim_min[i=net_data[:E]], κ[i] >= net_data[:κ̲][i])
    # gas flow equations
    @NLconstraint(model, w_eq[l=net_data[:E]], φ[l]*abs(φ[l]) - net_data[:k][l]^2 *(𝛑[ns(l)] + κ[l] - 𝛑[nr(l)]) == 0)
    @constraint(model, gas_bal, ϑ .- net_data[:δ] .- forecast[:ξ][:,s] .- net_data[:B]*κ .== net_data[:A]*φ)
    @constraint(model, φ_pl[l=net_data[:E_a]], φ[l] >= 0)
    @constraint(model, 𝛑[net_data[:ref]] == lin_res[:𝛑̇][net_data[:ref]])
    # solve model
    optimize!(model)
    # return solution
    solution = Dict(
    :Δϑ => sum(norm(JuMP.value.(ϑ)[n] - sol_ofs[:ϑ][n,s]) for n in N_ϑ),
    :Δκ => sum(sqrt(norm(JuMP.value.(κ)[n] - sol_ofs[:κ][n,s])) for n in N_κ),
    )
end
function projection_analysis(net_data,forecast,sol_ofs)
    """
    computes projection statistics
    """
    S = forecast[:S]
    Δϑ = zeros(S) # gas injection projection
    Δκ = zeros(S) # pressure regulation projection
    for s in 1:S
        sol_proj = projection_opt(net_data,forecast,sol_ofs,s)
        Δϑ = sol_proj[:Δϑ]
        Δκ = sol_proj[:Δκ]
    end
    return Dict(:Δϑ_mean => mean(Δϑ), :Δκ_mean => mean(Δκ))
end
