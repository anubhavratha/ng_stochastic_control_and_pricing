function non_convex_opt(net_data)
    """
    solves non-convex gas network optimization
    """
    # build model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(net_data[:N])])
    @variable(model, φ[1:length(net_data[:E])])
    @variable(model, κ[1:length(net_data[:E])])
    @variable(model, ϑ[1:length(net_data[:N])])
    # minimize gas production cost
    @objective(model, Min, ϑ'*diagm(vec(net_data[:c]))*ϑ)
    # gas variable limits
    @constraint(model, inj_lim_max[i=net_data[:N]], ϑ[i] <= net_data[:ϑ̅][i])
    @constraint(model, inj_lim_min[i=net_data[:N]], ϑ[i] >= net_data[:ϑ̲][i])
    @constraint(model, pre_lim_max[i=net_data[:N]], 𝛑[i] <= net_data[:ρ̅][i]^2)
    @constraint(model, pre_lim_min[i=net_data[:N]], 𝛑[i] >= net_data[:ρ̲][i]^2)
    @constraint(model, com_lim_max[i=net_data[:E]], κ[i] <= net_data[:κ̅][i])
    @constraint(model, com_lim_min[i=net_data[:E]], κ[i] >= net_data[:κ̲][i])
    # gas flow equations
    @NLconstraint(model, w_eq[l=net_data[:E]], φ[l]*abs(φ[l]) - net_data[:k][l]^2 *(𝛑[ns(l)] + κ[l] - 𝛑[nr(l)]) == 0)
    @constraint(model, gas_bal, ϑ .- net_data[:δ] .- net_data[:B]*κ .== net_data[:A]*φ)
    @constraint(model, φ_pl[l=net_data[:E_a]], φ[l] >= 0)
    # solve model
    optimize!(model)
    @info("non_convex model terminates with status: $(termination_status(model))")
    # return solution
    solution = Dict(
    :ϑ => JuMP.value.(ϑ),
    :𝛑 => JuMP.value.(𝛑),
    :φ => JuMP.value.(φ),
    :κ => JuMP.value.(κ),
    :cost => JuMP.objective_value.(model),
    :model => model)
    return solution
end

function linearized_opt(net_data,lin_res)
    """
    solves linearized gas network optimization
    """
    # build model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(net_data[:N])])
    @variable(model, φ[1:length(net_data[:E])])
    @variable(model, κ[1:length(net_data[:E])])
    @variable(model, ϑ[1:length(net_data[:N])])
    # minimize gas production cost
    @objective(model, Min, ϑ'*diagm(vec(net_data[:c]))*ϑ)
    # gas variable limits
    @constraint(model, inj_lim_max[i=net_data[:N]], ϑ[i] <= net_data[:ϑ̅][i])
    @constraint(model, inj_lim_min[i=net_data[:N]], ϑ[i] >= net_data[:ϑ̲][i])
    @constraint(model, pre_lim_max[i=net_data[:N]], 𝛑[i] <= net_data[:ρ̅][i]^2)
    @constraint(model, pre_lim_min[i=net_data[:N]], 𝛑[i] >= net_data[:ρ̲][i]^2)
    @constraint(model, com_lim_max[i=net_data[:E]], κ[i] <= net_data[:κ̅][i])
    @constraint(model, com_lim_min[i=net_data[:E]], κ[i] >= net_data[:κ̲][i])
    # gas flow equations
    @constraint(model, w_eq, φ .== lin_res[:ς1] + lin_res[:ς2] * 𝛑 + lin_res[:ς3] * κ)
    @constraint(model, gas_bal, ϑ .- net_data[:δ] .- net_data[:B]*κ .== net_data[:A]*φ)
    @constraint(model, φ_pl[l=net_data[:E_a]], φ[l] >= 0)
    @constraint(model, 𝛑[net_data[:ref]] == lin_res[:𝛑̇][net_data[:ref]])
    # solve model
    optimize!(model)
    @info("linearized model terminates with status: $(termination_status(model))")
    # return solution
    solution = Dict(
    :ϑ => JuMP.value.(ϑ),
    :𝛑 => JuMP.value.(𝛑),
    :φ => JuMP.value.(φ),
    :κ => JuMP.value.(κ),
    :cost => JuMP.objective_value.(model),
    :λ_c => JuMP.dual.(gas_bal),
    :λ_w => JuMP.dual.(w_eq),
    :model => model)
    return solution
end

function chance_con_opt(net_data,lin_res,forecast,settings)
    """
    solves chance-constrained gas network optimization
    """
    # build model
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(net_data[:N])])
    @variable(model, φ[1:length(net_data[:E])])
    @variable(model, κ[1:length(net_data[:E])])
    @variable(model, ϑ[1:length(net_data[:N])])
    @variable(model, α[1:length(net_data[:N]),1:length(net_data[:N])])
    @variable(model, β[1:length(net_data[:E]),1:length(net_data[:N])])
    @variable(model, s_𝛑[1:length(net_data[:N])])
    @variable(model, s_φ[1:length(net_data[:E])])
    @variable(model, c_ϑ[1:length(net_data[:N])])
    @variable(model, c_α[1:length(net_data[:N])])
    # minimize expected gas injection cost + variance penalty
    @objective(model, Min, sum(c_ϑ) + sum(c_α) + settings[:ψ_𝛑]*sum(s_𝛑) + settings[:ψ_φ]*sum(s_φ))
    # exp cost quadratic terms
    @constraint(model, λ_μ_u_ϑ[n=net_data[:N]], [1/2;c_ϑ[n];net_data[:c̀][n]*ϑ[n]] in RotatedSecondOrderCone())
    @constraint(model, λ_μ_u_α[n=net_data[:N]], [1/2;c_α[n];net_data[:c̀][n]*forecast[:Σ½]*α[n,:]] in RotatedSecondOrderCone())
    # gas flow equations
    @constraint(model, λ_c, 0 .== net_data[:A]*φ .- ϑ .+ net_data[:B]*κ .+ net_data[:δ])
    @constraint(model, λ_w, 0 .== φ .- lin_res[:ς1] .- lin_res[:ς2] * 𝛑 .- lin_res[:ς3] * κ)
    @constraint(model, λ_π, 𝛑[net_data[:ref]] == lin_res[:𝛑̇][net_data[:ref]])
    @constraint(model, λ_r, (α - net_data[:B]*β)'*ones(length(net_data[:N])) .== forecast[:I_δ])
    # # chance constraints
    # variance control
    @constraint(model, λ_u_s_π[n=net_data[:N]],  [s_𝛑[n] - 0; forecast[:Σ½] * (lin_res[:ς̆2]*(α .- lin_res[:ς̂3]*β .- diagm(ones(length(net_data[:N])))))[n,:]] in SecondOrderCone())
    @constraint(model, λ_u_s_φ[l=net_data[:E]],  [s_φ[l] - 0; forecast[:Σ½] * (lin_res[:ς̀2]*(α .- diagm(ones(length(net_data[:N])))) - lin_res[:ς̀3]*β)[l,:]] in SecondOrderCone())
    # pressure limits
    @constraint(model, λ_u_π̅[n=net_data[:N]],    [net_data[:ρ̅][n]^2 - 𝛑[n]; Φ(settings,net_data) * forecast[:Σ½] * (lin_res[:ς̆2]*(α .- lin_res[:ς̂3]*β .- diagm(ones(length(net_data[:N])))))[n,:]] in SecondOrderCone())
    @constraint(model, λ_u_π̲[n=net_data[:N]],    [𝛑[n] - net_data[:ρ̲][n]^2; Φ(settings,net_data) * forecast[:Σ½] * (lin_res[:ς̆2]*(α .- lin_res[:ς̂3]*β .- diagm(ones(length(net_data[:N])))))[n,:]] in SecondOrderCone())
    # flow limits
    @constraint(model, λ_u_φ̲[l=net_data[:E_a]],  [φ[l] - 0; Φ(settings,net_data) * forecast[:Σ½] * (lin_res[:ς̀2]*(α .- diagm(ones(length(net_data[:N])))) - lin_res[:ς̀3]*β)[l,:]] in SecondOrderCone())
    # injection limits
    @constraint(model, λ_u_ϑ̅[n=net_data[:N]],  [net_data[:ϑ̅][n] - ϑ[n]; Φ(settings,net_data) * forecast[:Σ½] * α[n,:]] in SecondOrderCone())
    @constraint(model, λ_u_ϑ̲[n=net_data[:N]],  [ϑ[n] - net_data[:ϑ̲][n]; Φ(settings,net_data) * forecast[:Σ½] * α[n,:]] in SecondOrderCone())
    # compression limits
    @constraint(model, λ_u_κ̅[l=net_data[:E]],  [net_data[:κ̅][l] - κ[l]; Φ(settings,net_data) * forecast[:Σ½] * β[l,:]] in SecondOrderCone())
    @constraint(model, λ_u_κ̲[l=net_data[:E]],  [κ[l] - net_data[:κ̲][l]; Φ(settings,net_data) * forecast[:Σ½] * β[l,:]] in SecondOrderCone())
    # aux constraints
    @constraint(model, δ̸_β, β[:,forecast[:N_δ̸]] .== 0)
    @constraint(model, δ̸_α, α[:,forecast[:N_δ̸]] .== 0)
    settings[:comp] == false ? @constraint(model, β[findall(x->x>0, net_data[:κ̅]),:] .==0) : NaN    # zero compressor response to uncertainty
    settings[:valv] == false ? @constraint(model, β[findall(x->x<0, net_data[:κ̲]),:] .==0) : NaN    # zero valve response to uncertainty
    # solve model
    optimize!(model)
    @info("policy optimization terminates with status: $(termination_status(model))")
    # return solution
    solution = Dict(
    # primal solution
    :obj => JuMP.objective_value.(model),
    :c_ϑ => JuMP.value.(c_ϑ),
    :c_α => JuMP.value.(c_α),
    :ϑ => JuMP.value.(ϑ),
    :𝛑 => JuMP.value.(𝛑),
    :φ => JuMP.value.(φ),
    :κ => JuMP.value.(κ),
    :α => JuMP.value.(α),
    :β => JuMP.value.(β),
    :s_𝛑 => JuMP.value.(s_𝛑),
    :s_φ => JuMP.value.(s_φ),
    :cost => JuMP.value.(ϑ)'*diagm(vec(net_data[:c]))*JuMP.value.(ϑ) + tr(JuMP.value.(α)'diagm(vec(net_data[:c]))JuMP.value.(α)*forecast[:Σ]),
    :λ_μ_u_ϑ => JuMP.dual.(λ_μ_u_ϑ),
    :λ_μ_u_α => JuMP.dual.(λ_μ_u_α),
    :λ_c => JuMP.dual.(λ_c),
    :λ_w => JuMP.dual.(λ_w),
    :λ_π => JuMP.dual.(λ_π),
    :λ_r => JuMP.dual.(λ_r),
    :λ_u_π̅ => JuMP.dual.(λ_u_π̅),
    :λ_u_π̲ => JuMP.dual.(λ_u_π̲),
    :λ_u_φ̲ => JuMP.dual.(λ_u_φ̲),
    :λ_u_s_π => JuMP.dual.(λ_u_s_π),
    :λ_u_s_φ => JuMP.dual.(λ_u_s_φ),
    :λ_u_ϑ̅ => JuMP.dual.(λ_u_ϑ̅),
    :λ_u_ϑ̲ => JuMP.dual.(λ_u_ϑ̲),
    :λ_u_κ̅ => JuMP.dual.(λ_u_κ̅),
    :λ_u_κ̲ => JuMP.dual.(λ_u_κ̲),
    :model => model)
    return solution
end

function stochastic_dual_solution(net_data,sol_stochastic,lin_res,forecast,settings)
    """
    extracts and processes the dual solution of the chance-constrained gas network optimization
    """
    # extract dual variables
    λ_c = sol_stochastic[:λ_c]
    λ_w = sol_stochastic[:λ_w]
    λ_π̇ = sol_stochastic[:λ_π]
    λ_r = sol_stochastic[:λ_r]
    λ_ϑ = [sol_stochastic[:λ_μ_u_ϑ][i][1] for i in net_data[:N]]
    λ_α = [sol_stochastic[:λ_μ_u_α][i][1] for i in net_data[:N]]
    λ_π̅ = [sol_stochastic[:λ_u_π̅][i][1] for i in net_data[:N]]
    λ_π̲ = [sol_stochastic[:λ_u_π̲][i][1] for i in net_data[:N]]
    λ_ϑ̅ = [sol_stochastic[:λ_u_ϑ̅][i][1] for i in net_data[:N]]
    λ_ϑ̲ = [sol_stochastic[:λ_u_ϑ̲][i][1] for i in net_data[:N]]
    λ_κ̅ = [sol_stochastic[:λ_u_κ̅][i][1] for i in net_data[:E]]
    λ_κ̲ = [sol_stochastic[:λ_u_κ̲][i][1] for i in net_data[:E]]
    λ_s_π = [sol_stochastic[:λ_u_s_π][i][1] for i in net_data[:N]]
    λ_s_φ = [sol_stochastic[:λ_u_s_φ][i][1] for i in net_data[:E]]
    u_ϑ = [sol_stochastic[:λ_μ_u_ϑ][i][3] for i in net_data[:N]]
    u_α = [sol_stochastic[:λ_μ_u_α][i][j] for i in net_data[:N], j in 3:length(net_data[:N])+2]
    u_π̅ = [sol_stochastic[:λ_u_π̅][i][j] for i in net_data[:N], j in 2:length(net_data[:N])+1]
    u_π̲ = [sol_stochastic[:λ_u_π̲][i][j] for i in net_data[:N], j in 2:length(net_data[:N])+1]
    u_ϑ̅ = [sol_stochastic[:λ_u_ϑ̅][i][j] for i in net_data[:N], j in 2:length(net_data[:N])+1]
    u_ϑ̲ = [sol_stochastic[:λ_u_ϑ̲][i][j] for i in net_data[:N], j in 2:length(net_data[:N])+1]
    u_κ̅ = [sol_stochastic[:λ_u_κ̅][i][j] for i in net_data[:E], j in 2:length(net_data[:N])+1]
    u_κ̲ = [sol_stochastic[:λ_u_κ̲][i][j] for i in net_data[:E], j in 2:length(net_data[:N])+1]
    u_s_π = [sol_stochastic[:λ_u_s_π][i][j] for i in net_data[:N], j in 2:length(net_data[:N])+1]
    u_s_φ = [sol_stochastic[:λ_u_s_φ][i][j] for i in net_data[:E], j in 2:length(net_data[:N])+1]
    λ_φ̲ = zeros(length(net_data[:E]))
    u_φ̲ = zeros(length(net_data[:E]),length(net_data[:N]))
    for i in net_data[:E]
        i ∈ net_data[:E_a] ? λ_φ̲[i] = sol_stochastic[:λ_u_φ̲][i][1] : NaN
        i ∈ net_data[:E_a] ? u_φ̲[i,:] = sol_stochastic[:λ_u_φ̲][i][2:end] : NaN
    end
    μ_ϑ = [sol_stochastic[:λ_μ_u_ϑ][i][2] for i in net_data[:N]]
    μ_α =  [sol_stochastic[:λ_μ_u_α][i][2] for i in net_data[:N]]

    # partial lagrangian function
    L_part = (
    + λ_c'*(net_data[:A]*sol_stochastic[:φ] .- sol_stochastic[:ϑ] + net_data[:B]*sol_stochastic[:κ] + net_data[:δ])
    + λ_r'*(ones(length(net_data[:N])) - (sol_stochastic[:α]-net_data[:B]*sol_stochastic[:β])'*ones(length(net_data[:N])))
    + λ_w'*(sol_stochastic[:φ] - lin_res[:ς1] - lin_res[:ς2]*sol_stochastic[:𝛑] - lin_res[:ς3]*sol_stochastic[:κ])
    - λ_s_φ'*sol_stochastic[:s_φ]
    - λ_s_π'*sol_stochastic[:s_𝛑]
    - λ_φ̲'*sol_stochastic[:φ]
    - λ_π̅'*(net_data[:ρ̅].^2 - sol_stochastic[:𝛑])
    - λ_π̲'*(sol_stochastic[:𝛑] - net_data[:ρ̲].^2)
    - sum((u_s_φ + Φ(settings,net_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*(sol_stochastic[:α] - diagm(ones(length(net_data[:N])))) - lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in net_data[:E])
    - sum((u_s_π + Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*(sol_stochastic[:α] - lin_res[:ς̂3]*sol_stochastic[:β] - diagm(ones(length(net_data[:N])))))[n,:] for n in net_data[:N])
    )

    # partial lagrangian function decomposition
    R_free = + λ_w' * lin_res[:ς1]
    R_rent = (
            - λ_c'*net_data[:A] * sol_stochastic[:φ]
            - λ_w' * sol_stochastic[:φ]
            + λ_w' * lin_res[:ς2] * sol_stochastic[:𝛑]
            + λ_φ̲'*sol_stochastic[:φ]
            + λ_π̅'*(net_data[:ρ̅].^2 - sol_stochastic[:𝛑])
            + λ_π̲'*(sol_stochastic[:𝛑] - net_data[:ρ̲].^2)
            + λ_s_φ'*sol_stochastic[:s_φ]
            + λ_s_π'*sol_stochastic[:s_𝛑]
            )
    R_inj = (
            + λ_c' * sol_stochastic[:ϑ]
            + λ_r' * sol_stochastic[:α]'*ones(length(net_data[:N]))
            + sum((u_s_φ + Φ(settings,net_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*sol_stochastic[:α])[l,:] for l in net_data[:E])
            + sum((u_s_π + Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*sol_stochastic[:α])[n,:] for n in net_data[:N])
            )
    R_act = (
            - λ_c' * net_data[:B] * sol_stochastic[:κ]
            - λ_r' * (net_data[:B]*sol_stochastic[:β])' * ones(length(net_data[:N]))
            + λ_w' * lin_res[:ς3] * sol_stochastic[:κ]
            - sum((u_s_φ + Φ(settings,net_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in net_data[:E])
            - sum((u_s_π + Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*lin_res[:ς̂3]*sol_stochastic[:β])[n,:] for n in net_data[:N])
            )
    R_con = (
            + λ_c' * net_data[:δ]
            + λ_r' * ones(length(net_data[:N]))
            + sum((u_s_φ + Φ(settings,net_data)*u_φ̲)[l,:]'*forecast[:Σ½]*lin_res[:ς̀2][l,:] for l in net_data[:E])
            + sum((u_s_π + Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*lin_res[:ς̆2][n,:] for n in net_data[:N])
            )

    # revenue balance
    R_con - R_inj - R_act - R_rent - R_free
    # revenue decomposition
    R_inj_nom_bal = λ_c' * sol_stochastic[:ϑ]
    R_inj_rec_bal = λ_r' * sol_stochastic[:α]'*ones(length(net_data[:N]))
    R_inj_net_lim = sum(Φ(settings,net_data)*u_φ̲[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*sol_stochastic[:α])[l,:] for l in net_data[:E]) + sum((Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*sol_stochastic[:α])[n,:] for n in net_data[:N])
    R_inj_net_var = sum(u_s_φ[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*sol_stochastic[:α])[l,:] for l in net_data[:E]) + sum(u_s_π[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*sol_stochastic[:α])[n,:] for n in net_data[:N])

    R_act_nom_bal = λ_w' * lin_res[:ς3] * sol_stochastic[:κ] - λ_c' * net_data[:B] * sol_stochastic[:κ]
    R_act_rec_bal = - λ_r' * (net_data[:B]*sol_stochastic[:β])' * ones(length(net_data[:N]))
    R_act_net_lim = - sum((Φ(settings,net_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in net_data[:E]) - sum((Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*lin_res[:ς̂3]*sol_stochastic[:β])[n,:] for n in net_data[:N])
    R_act_net_var = - sum(u_s_φ[l,:]'*forecast[:Σ½]*(lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in net_data[:E]) - sum(u_s_π[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*lin_res[:ς̂3]*sol_stochastic[:β])[n,:] for n in net_data[:N])

    R_con_nom_bal = λ_c' * net_data[:δ]
    R_con_rec_bal = λ_r' * ones(length(net_data[:N]))
    R_con_net_lim = sum((Φ(settings,net_data)*u_φ̲)[l,:]'*forecast[:Σ½]*lin_res[:ς̀2][l,:] for l in net_data[:E]) + sum((Φ(settings,net_data)*u_π̅ + Φ(settings,net_data)*u_π̲)[n,:]' * forecast[:Σ½]*lin_res[:ς̆2][n,:] for n in net_data[:N])
    R_con_net_var = sum(u_s_φ[l,:]'*forecast[:Σ½]*lin_res[:ς̀2][l,:] for l in net_data[:E]) + sum(u_s_π[n,:]' * forecast[:Σ½]*lin_res[:ς̆2][n,:] for n in net_data[:N])

    Revenue_decomposition = DataFrame(source = ["nom_bal","rec_bal","net_lim","net_var","total"],
                                            inj = [R_inj_nom_bal,R_inj_rec_bal,R_inj_net_lim,R_inj_net_var, R_inj_nom_bal+R_inj_rec_bal+R_inj_net_lim+R_inj_net_var],
                                            act = [R_act_nom_bal,R_act_rec_bal,R_act_net_lim,R_act_net_var, R_act_nom_bal+R_act_rec_bal+R_act_net_lim+R_act_net_var],
                                            con = [R_con_nom_bal,R_con_rec_bal,R_con_net_lim,R_con_net_var, R_con_nom_bal+R_con_rec_bal+R_con_net_lim+R_con_net_var]
    )

    # compute indvidual revenues & profits
    R_ind_inj = [λ_c[n]*sol_stochastic[:ϑ][n] + (λ_r' + lin_res[:ς̆2][:,n]'*(u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲)*forecast[:Σ½] + lin_res[:ς̀2][:,n]'*(u_s_φ .+ Φ(settings,net_data) * u_φ̲)*forecast[:Σ½])*sol_stochastic[:α][n,:] for n in net_data[:N]]
    Π_inj = [R_ind_inj[n] - sol_stochastic[:c_ϑ][n] - sol_stochastic[:c_α][n] for n in net_data[:N]]
    R_ind_act = [ -λ_c'*net_data[:B][:,l] * sol_stochastic[:κ][l] + lin_res[:ς3][:,l]' * λ_w * sol_stochastic[:κ][l] - ones(length(net_data[:N]))' * net_data[:B][:,l] * λ_r' * sol_stochastic[:β][l,:] - (lin_res[:ς̆2]*lin_res[:ς̂3])[:,l]' * (u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲) * forecast[:Σ½] * sol_stochastic[:β][l,:] - lin_res[:ς̀3][:,l]' * (u_s_φ .+ Φ(settings,net_data) * u_φ̲) * forecast[:Σ½] * sol_stochastic[:β][l,:] for l in net_data[:E_a]]
    R_ind_rent = (- λ_c'*net_data[:A] * sol_stochastic[:φ] - λ_w' * sol_stochastic[:φ] + λ_w' * lin_res[:ς2] * sol_stochastic[:𝛑] + λ_φ̲'*sol_stochastic[:φ] + λ_π̅'*(net_data[:ρ̅].^2 - sol_stochastic[:𝛑]) + λ_π̲'*(sol_stochastic[:𝛑] - net_data[:ρ̲].^2) + λ_s_φ'*sol_stochastic[:s_φ] + λ_s_π'*sol_stochastic[:s_𝛑])
    # R_ind_con = [λ_c[n]*net_data[:δ][n] + λ_r[n] + sum(lin_res[:ς̀2][l,n]*(u_s_φ .+ Φ(settings,net_data) * u_φ̲)[l,:]'*forecast[:Σ½][n,:] for l in net_data[:E]) + sum(lin_res[:ς̆2][k,n]*(u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲)[k,:]'*forecast[:Σ½][n,:] for k in net_data[:N]) for n in net_data[:N]]
    R_ind_con = [λ_c[n]*net_data[:δ][n] + λ_r[n] + forecast[:Σ½][n,:]'*(u_s_φ .+ Φ(settings,net_data) * u_φ̲)'*lin_res[:ς̀2][:,n] + forecast[:Σ½][n,:]'*(u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲)'*lin_res[:ς̆2][:,n] for n in net_data[:N]]

    # check stationarity conditions
    ∂L_∂s_φ = settings[:ψ_φ] .- λ_s_φ
    ∂L_∂s_π = settings[:ψ_𝛑] .- λ_s_π
    ∂L_∂c_ϑ = 1 .- μ_ϑ
    ∂L_∂c_α = 1 .- μ_α
    ∂L_∂π = λ_π̅ .- λ_π̲ .- vec(λ_w'*lin_res[:ς2]) ; ∂L_∂π[net_data[:ref]] = ∂L_∂π[net_data[:ref]] - λ_π̇
    ∂L_∂φ = vec(λ_c'*net_data[:A]) .+ λ_w .- λ_φ̲
    ∂L_∂ϑ = - u_ϑ .* net_data[:c̀] .- λ_c .+ λ_ϑ̅ .- λ_ϑ̲
    ∂L_∂κ = vec(λ_c'*net_data[:B]) .- vec(λ_w'*lin_res[:ς3]) .+ λ_κ̅ .- λ_κ̲
    ∂L_∂α = zeros(length(net_data[:N]),length(net_data[:N]))
    ∂L_∂β = zeros(length(net_data[:E]),length(net_data[:N]))
    for n in net_data[:N]
        # ∂L_∂α[n,:] = 2*net_data[:c][n]*forecast[:Σ]*sol_stochastic[:α][n,:] .- λ_r .- forecast[:Σ½] * (u_φ' * lin_res[:ς̀2][:,n] + u_π' * lin_res[:ς̆2][:,n] + Φ(settings,net_data) * u_ϑ̅[n,:] + Φ(settings,net_data) * u_ϑ̲[n,:])
        ∂L_∂α[n,:] = -forecast[:Σ½]*u_α[n,:]*net_data[:c̀][n] .- λ_r .- forecast[:Σ½] * ((u_s_φ .+ Φ(settings,net_data) * u_φ̲)' * lin_res[:ς̀2][:,n] + (u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲)' * lin_res[:ς̆2][:,n] + Φ(settings,net_data) * u_ϑ̅[n,:] + Φ(settings,net_data) * u_ϑ̲[n,:])
    end
    for l in net_data[:E]
        ∂L_∂β[l,:] = ones(length(net_data[:N]))'*net_data[:B][:,l]*λ_r .+ forecast[:Σ½] * (u_s_φ .+ Φ(settings,net_data) * u_φ̲)' * lin_res[:ς̀3][:,l] .+ forecast[:Σ½] * (u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲)' * (lin_res[:ς̆2]*lin_res[:ς̂3])[:,l] .- forecast[:Σ½] * Φ(settings,net_data) * (u_κ̅ + u_κ̲)[l,:]
    end
    # total stationarity conditions mismatch
    mismatch = (sum(abs.(∂L_∂c_ϑ)) + sum(abs.(∂L_∂c_α)) + sum(abs.(∂L_∂s_φ))
                + sum(abs.(∂L_∂s_π)) + sum(abs.(∂L_∂π)) + sum(abs.(∂L_∂φ))
                + sum(abs.(∂L_∂ϑ)) + sum(abs.(∂L_∂κ)) + sum(abs.(∂L_∂β)))
    mismatch <= 1e-2 ? @info("stationarity conditions hold; total mismatch : $(round(mismatch,digits=8))") : @warn("stationarity conditions do not hold;  mismatch : $(round(mismatch,digits=8))")

    # compute dual objective function
    dual_obj = (λ_c' * net_data[:δ]
    + sum(λ_r)
    - λ_w' * lin_res[:ς1]
    - 0.5 * sum(λ_ϑ)
    - 0.5 * sum(λ_α)
    - λ_π̅' * net_data[:ρ̅].^2
    + λ_π̲' * net_data[:ρ̲].^2
    - λ_ϑ̅' * net_data[:ϑ̅]
    + λ_ϑ̲' * net_data[:ϑ̲]
    - λ_κ̅' * net_data[:κ̅]
    + λ_κ̲' * net_data[:κ̲]
    + sum((u_s_φ .+ Φ(settings,net_data) * u_φ̲)[l,:]' * forecast[:Σ½] * (lin_res[:ς̀2] * diagm(ones(length(net_data[:N]))))[l,:] for l in net_data[:E])
    + sum((u_s_π .+ Φ(settings,net_data) * u_π̅ .+ Φ(settings,net_data)*  u_π̲)[n,:]' * forecast[:Σ½] * (lin_res[:ς̆2] * diagm(ones(length(net_data[:N]))))[n,:] for n in net_data[:N])
    + λ_π̇*lin_res[:𝛑̇][net_data[:ref]])
    # check duality gap
    duality_gap = norm(dual_obj-sol_stochastic[:obj])/sol_stochastic[:obj]*100
    duality_gap <= 1e-3 ? @info("strong duality holds; duality gap : $(round(duality_gap,digits=3))%") : @info("strong duality does not hold; duality gap : $(round(duality_gap,digits=3))%")

    # return dual solution
    return Dict(:dual_obj => dual_obj, :R_inj => round.(R_inj), :Π_inj => round.(Π_inj), :R_act => round.(R_act), :R_con => round.(R_con), :R_rent => round.(R_rent), :Revenue_decomposition => Revenue_decomposition)
end
