using CSV, DataFrames
using JuMP, Ipopt
using LinearAlgebra
using SparseArrays
using Mosek, MosekTools
using Distributions
using StatsBase

function load_data(case)
    """
    extract, transform and load case data from csv files
    """
    prod_data = DataFrame!(CSV.File("data/$(case)/gas_prod.csv"))
    node_data = DataFrame!(CSV.File("data/$(case)/gas_node.csv"))
    pipe_data = DataFrame!(CSV.File("data/$(case)/gas_pipe.csv"))
    # sets
    E  = collect(1:length(pipe_data[!,:k]))
    E_a = vcat(findall(x->x>0, pipe_data[!,:kappa_max]), findall(x->x<0, pipe_data[!,:kappa_min]))
    N  = collect(1:length(node_data[!,:demand]))
    # gas producer data
    c = Array(prod_data[!,:c])
    c̀ = sqrt.(c)
    ϑ̅ = Array(prod_data[!,:p_max])
    ϑ̲ = Array(prod_data[!,:p_min])
    # node data
    δ = Array(node_data[!,:demand])
    ρ̅ = Array(node_data[!,:presh_max])
    ρ̲ = Array(node_data[!,:presh_min])
    # edge data
    n_s = Array(pipe_data[!,:n_s])
    n_r = Array(pipe_data[!,:n_r])
    k = Array(pipe_data[!,:k])
    κ̅ = Array(pipe_data[!,:kappa_max])
    κ̲ = Array(pipe_data[!,:kappa_min])
    # reference pressure node
    ref = 26
    # node-edge incidence matrix
    A = zeros(size(N,1),size(E,1))
    for i in 1:size(E,1), j in 1:size(N,1)
        j == n_s[i] ? A[j,i] = 1 : NaN
        j == n_r[i] ? A[j,i] = -1 : NaN
    end
    # gas - pressure conversion matrix
    B = zeros(size(N,1),size(E,1))
    for j in E
        if j ∈ E_a
            pipe_data[j,:kappa_max] > 0 ? B[n_s[j],j] = 0.00005 : NaN
            pipe_data[j,:kappa_min] < 0 ? B[n_s[j],j] = -0.00005 : NaN
        end
    end
    # number of operational constraints
    num_con = 2*length(N) + 2*length(findall(x->x>0, ϑ̅[:])) + 2*length(E) + length(E_a)
    # save gas system data
    gas_data = Dict(:c => c, :c̀ => c̀, :ϑ̅ => ϑ̅, :ϑ̲ => ϑ̲, :δ => δ, :ρ̅ => ρ̅, :ρ̲ => ρ̲,
                    :n_s => n_s, :n_r => n_r, :k => k, :κ̅ => κ̅, :κ̲ => κ̲, :A => A, :B => B, :ref => ref,
                    :num_con => num_con, :E => E, :E_a => E_a, :N => N)
    return gas_data
end

# auxiliary functions
ns(l) = Int(gas_data[:n_s][l])          #retrieve sending node of a pipeline
nr(l) = Int(gas_data[:n_r][l])          #retrieve receiving node of a pipeline

function Φ(settings,gas_data)
    """
    compute the safety factor
    """
    settings[:det] == true ? res = 0 : NaN
    settings[:det] == false ? res = quantile(Normal(0,1), 1 - settings[:ε]/gas_data[:num_con]) : NaN
    return res
end

function remove_col_and_row(A,ref)
    """
    get reduced matrix by removing row and col corresponding to reference node
    """
    @assert size(A,1) == size(A,2)
    n = size(A,1)
    return A[1:n .!= ref, 1:n .!= ref]
end
function full_matrix(A,ref)
    """
    retrieve a full matrix from the reduced matrix
    """
    Nb = size(A,1)+1
    V = zeros(Nb,Nb)
    for i in 1:Nb, j in 1:Nb
        i < ref && j < ref ? V[i,j] = A[i,j] : NaN
        i > ref && j > ref ? V[i,j] = A[i-1,j-1] : NaN
        i > ref && j < ref ? V[i,j] = A[i-1,j] : NaN
        i < ref && j > ref ? V[i,j] = A[i,j-1] : NaN
    end
    return V
end

function gas_non_convex(gas_data)
    """
    non-convex gas network optimization
    """
    # build model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(gas_data[:N])])
    @variable(model, φ[1:length(gas_data[:E])])
    @variable(model, κ[1:length(gas_data[:E])])
    @variable(model, ϑ[1:length(gas_data[:N])])
    # minimize gas production cost
    @objective(model, Min, ϑ'*diagm(vec(gas_data[:c]))*ϑ)
    # gas variable limits
    @constraint(model, inj_lim_max[i=gas_data[:N]], ϑ[i] <= gas_data[:ϑ̅][i])
    @constraint(model, inj_lim_min[i=gas_data[:N]], ϑ[i] >= gas_data[:ϑ̲][i])
    @constraint(model, pre_lim_max[i=gas_data[:N]], 𝛑[i] <= gas_data[:ρ̅][i]^2)
    @constraint(model, pre_lim_min[i=gas_data[:N]], 𝛑[i] >= gas_data[:ρ̲][i]^2)
    @constraint(model, com_lim_max[i=gas_data[:E]], κ[i] <= gas_data[:κ̅][i])
    @constraint(model, com_lim_min[i=gas_data[:E]], κ[i] >= gas_data[:κ̲][i])
    # gas flow equations
    @NLconstraint(model, w_eq[l=gas_data[:E]], φ[l]*abs(φ[l]) - gas_data[:k][l]^2 *(𝛑[ns(l)] + κ[l] - 𝛑[nr(l)]) == 0)
    @constraint(model, gas_bal, ϑ .- gas_data[:δ] .- gas_data[:B]*κ .== gas_data[:A]*φ)
    @constraint(model, φ_pl[l=gas_data[:E_a]], φ[l] >= 0)
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

function linearization(gas_data,model)
    """
    extract gas flow and pressure senstivities around operating point
    """
    con_nl = model[:w_eq]
    var_nl = [model[:𝛑];model[:φ];model[:κ]]
    # aux functions
    raw_index(var::MOI.VariableIndex) = var.value
    raw_index(con::NonlinearConstraintIndex) = con.value
    # extract variables
    var = all_variables(model)
    Nv = length(var)
    # extract Jacobian structure
    d = NLPEvaluator(model)
    MOI.initialize(d, [:Jac])
    jac_str = MOI.jacobian_structure(d)
    # extract operating point
    opertaing_point = value.(var)
    # evaluate Weymouth eq. Jacobian at the operating point
    V = zeros(length(jac_str))
    MOI.eval_constraint_jacobian(d, V, opertaing_point)
    # prepare the result
    I = [js[1] for js in jac_str] # rows, equations
    J = [js[2] for js in jac_str] # cols, variables
    jac = sparse(I, J, V)
    rows = raw_index.(index.(con_nl))
    cols = raw_index.(index.(var_nl))
    # Jacobian
    Jac = Matrix(jac[rows, cols])
    Jac_π = Jac[gas_data[:E],gas_data[:N]]
    Jac_φ = Jac[gas_data[:E],length(gas_data[:N]) .+ gas_data[:E]]
    Jac_κ = Jac[gas_data[:E],(length(gas_data[:N]) .+ length(gas_data[:E])) .+ gas_data[:E]]
    # operating point
    𝛑̇ = opertaing_point[raw_index.(index.(var_nl))][gas_data[:N]]
    φ̇ = opertaing_point[raw_index.(index.(var_nl))][length(gas_data[:N]) .+ gas_data[:E]]
    κ̇ = opertaing_point[raw_index.(index.(var_nl))][(length(gas_data[:N]) .+ length(gas_data[:E])) .+ gas_data[:E]]
    # linearization coefficients
    ς1 = inv(Jac_φ) * (Jac_π * 𝛑̇ + Jac_κ * κ̇ + Jac_φ * φ̇)
    ς2 = -inv(Jac_φ) * Jac_π
    ς3 = -inv(Jac_φ) * Jac_κ
    # pressure-related
    ς̂2 = gas_data[:A]*ς2
    ς̂3 = gas_data[:B] + gas_data[:A]*ς3
    ς̆2 = remove_col_and_row(ς̂2,gas_data[:ref])
    ς̆2 = full_matrix(inv(ς̆2),gas_data[:ref])
    # # flow-related
    ς̀2 = ς2*ς̆2
    ς̀3 = ς2*ς̆2*ς̂3 - ς3
    # save linearization results
    lin_res = Dict( :jac => Jac,
                    :𝛑̇ => 𝛑̇,
                    :φ̇ => φ̇,
                    :κ̇ => κ̇,
                    :ς1 => round.(ς1, digits = 10),
                    :ς2 => round.(ς2, digits = 10),
                    :ς3 => round.(ς3, digits = 10),
                    :ς̂2 => round.(ς̂2, digits = 10),
                    :ς̂3 => round.(ς̂3, digits = 10),
                    :ς̆2 => round.(ς̆2, digits = 10),
                    :ς̀2 => round.(ς̀2, digits = 10),
                    :ς̀3 => round.(ς̀3, digits = 10)
                    )
    maximum(-inv(Jac_φ) * Jac_π) >= 1e6 ? @info("most likely you are at Bifurcation point") : NaN
    maximum(-inv(Jac_φ) * Jac_κ) >= 1e6 ? @info("most likely you are at Bifurcation point") : NaN
    # solve linearized gas network optimization
    sol_linearized = gas_linearized(gas_data,lin_res)
    # check lienarization quality
    maximum(sol_non_convex[:𝛑] .- sol_linearized[:𝛑]) <= 1 ? @info("linearization successful") : @warn("linearization fails")
    return  lin_res
end

function gas_linearized(gas_data,lin_res)
    """
    linearized gas network optimization
    """
    # build model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(gas_data[:N])])
    @variable(model, φ[1:length(gas_data[:E])])
    @variable(model, κ[1:length(gas_data[:E])])
    @variable(model, ϑ[1:length(gas_data[:N])])
    # minimize gas production cost
    @objective(model, Min, ϑ'*diagm(vec(gas_data[:c]))*ϑ)
    # gas variable limits
    @constraint(model, inj_lim_max[i=gas_data[:N]], ϑ[i] <= gas_data[:ϑ̅][i])
    @constraint(model, inj_lim_min[i=gas_data[:N]], ϑ[i] >= gas_data[:ϑ̲][i])
    @constraint(model, pre_lim_max[i=gas_data[:N]], 𝛑[i] <= gas_data[:ρ̅][i]^2)
    @constraint(model, pre_lim_min[i=gas_data[:N]], 𝛑[i] >= gas_data[:ρ̲][i]^2)
    @constraint(model, com_lim_max[i=gas_data[:E]], κ[i] <= gas_data[:κ̅][i])
    @constraint(model, com_lim_min[i=gas_data[:E]], κ[i] >= gas_data[:κ̲][i])
    # gas flow equations
    @constraint(model, w_eq, φ .== lin_res[:ς1] + lin_res[:ς2] * 𝛑 + lin_res[:ς3] * κ)
    @constraint(model, gas_bal, ϑ .- gas_data[:δ] .- gas_data[:B]*κ .== gas_data[:A]*φ)
    @constraint(model, φ_pl[l=gas_data[:E_a]], φ[l] >= 0)
    @constraint(model, 𝛑[gas_data[:ref]] == lin_res[:𝛑̇][gas_data[:ref]])
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

function forecast_data(gas_data,settings)
    """
    get covariance matrix and its factorization, generate samples
    """
    N = length(gas_data[:N])
    N_δ = findall(x->x!=0, gas_data[:δ])
    N_δ̸ = setdiff(1:N,N_δ)
    I_δ = zeros(N); I_δ[N_δ] .= 1;

    Σ  = zeros(N,N)                     # extended covariance matrix
    Σ½ = zeros(N,N)                     # extended Cholesky factorization

    σ = zeros(N)                        # vector of standard deviations

    c = 0.00                            # correlation coefficient
    C = zeros(N,N)                      # correlation matrix

    for k in N_δ, j in N_δ
        σ[k] = settings[:σ]*gas_data[:δ][k]
        k != j ? C[k,j] = c : NaN
        k == j ? C[k,j] = 1 : NaN
    end

    Σ = cor2cov(C, σ)
    F = cholesky(Σ[N_δ,N_δ]).L          # Cholesky factorization: F*F' = Σ[N_δ,N_δ]
    Σ½[N_δ,N_δ] = F

    S = 10000
    ξ = zeros(N,S)
    ξ[N_δ,:] = rand(MvNormal(zeros(length(N_δ)),Σ[N_δ,N_δ]),S)

    return forecast = Dict(:Σ => Σ, :Σ½ => Σ½, :ξ => ξ, :σ => σ, :I_δ => I_δ, :N_δ => N_δ, :N_δ̸ => N_δ̸, :S => S)
end

function gas_cc(gas_data,lin_res,forecast,settings)
    """
    chance-constrained gas network optimization model
    """
    # build model
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(gas_data[:N])])
    @variable(model, φ[1:length(gas_data[:E])])
    @variable(model, κ[1:length(gas_data[:E])])
    @variable(model, ϑ[1:length(gas_data[:N])])
    @variable(model, α[1:length(gas_data[:N]),1:length(gas_data[:N])])
    @variable(model, β[1:length(gas_data[:E]),1:length(gas_data[:N])])
    @variable(model, s_𝛑[1:length(gas_data[:N])])
    @variable(model, s_φ[1:length(gas_data[:E])])
    @variable(model, c_ϑ[1:length(gas_data[:N])])
    @variable(model, c_α[1:length(gas_data[:N])])
    # minimize expected gas injection cost + variance penalty
    @objective(model, Min, sum(c_ϑ) + sum(c_α) + settings[:ψ_𝛑]*sum(s_𝛑) + settings[:ψ_φ]*sum(s_φ))
    # exp cost quadratic terms
    @constraint(model, λ_μ_u_ϑ[n=gas_data[:N]], [1/2;c_ϑ[n];gas_data[:c̀][n]*ϑ[n]] in RotatedSecondOrderCone())
    @constraint(model, λ_μ_u_α[n=gas_data[:N]], [1/2;c_α[n];gas_data[:c̀][n]*forecast[:Σ½]*α[n,:]] in RotatedSecondOrderCone())
    # gas flow equations
    @constraint(model, λ_c, 0 .== gas_data[:A]*φ .- ϑ .+ gas_data[:B]*κ .+ gas_data[:δ])
    @constraint(model, λ_w, 0 .== φ .- lin_res[:ς1] .- lin_res[:ς2] * 𝛑 .- lin_res[:ς3] * κ)
    @constraint(model, λ_π, 𝛑[gas_data[:ref]] == lin_res[:𝛑̇][gas_data[:ref]])
    @constraint(model, λ_r, (α - gas_data[:B]*β)'*ones(length(gas_data[:N])) .== forecast[:I_δ])
    # # chance constraints
    # variance control
    @constraint(model, λ_u_s_π[n=gas_data[:N]],  [s_𝛑[n] - 0; forecast[:Σ½] * (lin_res[:ς̆2]*(α .- lin_res[:ς̂3]*β .- diagm(ones(length(gas_data[:N])))))[n,:]] in SecondOrderCone())
    @constraint(model, λ_u_s_φ[l=gas_data[:E]],  [s_φ[l] - 0; forecast[:Σ½] * (lin_res[:ς̀2]*(α .- diagm(ones(length(gas_data[:N])))) - lin_res[:ς̀3]*β)[l,:]] in SecondOrderCone())
    # pressure limits
    @constraint(model, λ_u_π̅[n=gas_data[:N]],    [gas_data[:ρ̅][n]^2 - 𝛑[n]; Φ(settings,gas_data) * forecast[:Σ½] * (lin_res[:ς̆2]*(α .- lin_res[:ς̂3]*β .- diagm(ones(length(gas_data[:N])))))[n,:]] in SecondOrderCone())
    @constraint(model, λ_u_π̲[n=gas_data[:N]],    [𝛑[n] - gas_data[:ρ̲][n]^2; Φ(settings,gas_data) * forecast[:Σ½] * (lin_res[:ς̆2]*(α .- lin_res[:ς̂3]*β .- diagm(ones(length(gas_data[:N])))))[n,:]] in SecondOrderCone())
    # flow limits
    @constraint(model, λ_u_φ̲[l=gas_data[:E_a]],  [φ[l] - 0; Φ(settings,gas_data) * forecast[:Σ½] * (lin_res[:ς̀2]*(α .- diagm(ones(length(gas_data[:N])))) - lin_res[:ς̀3]*β)[l,:]] in SecondOrderCone())
    # injection limits
    @constraint(model, λ_u_ϑ̅[n=gas_data[:N]],  [gas_data[:ϑ̅][n] - ϑ[n]; Φ(settings,gas_data) * forecast[:Σ½] * α[n,:]] in SecondOrderCone())
    @constraint(model, λ_u_ϑ̲[n=gas_data[:N]],  [ϑ[n] - gas_data[:ϑ̲][n]; Φ(settings,gas_data) * forecast[:Σ½] * α[n,:]] in SecondOrderCone())
    # compression limits
    @constraint(model, λ_u_κ̅[l=gas_data[:E]],  [gas_data[:κ̅][l] - κ[l]; Φ(settings,gas_data) * forecast[:Σ½] * β[l,:]] in SecondOrderCone())
    @constraint(model, λ_u_κ̲[l=gas_data[:E]],  [κ[l] - gas_data[:κ̲][l]; Φ(settings,gas_data) * forecast[:Σ½] * β[l,:]] in SecondOrderCone())
    # aux constraints
    @constraint(model, δ̸_β, β[:,forecast[:N_δ̸]] .== 0)
    @constraint(model, δ̸_α, α[:,forecast[:N_δ̸]] .== 0)
    # solve model
    optimize!(model)
    @info("stochastic model terminates with status: $(termination_status(model))")
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
    :cost => JuMP.value.(ϑ)'*diagm(vec(gas_data[:c]))*JuMP.value.(ϑ) + tr(JuMP.value.(α)'diagm(vec(gas_data[:c]))JuMP.value.(α)*forecast[:Σ]),
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

function out_of_sample(gas_data,forecast,sol_stochastic)
    """
    perform out-of-sample simulations
    """
    φ̃ = zeros(length(gas_data[:E]),forecast[:S])
    ϑ̃ = zeros(length(gas_data[:N]),forecast[:S])
    κ̃ = zeros(length(gas_data[:E]),forecast[:S])
    π̃ = zeros(length(gas_data[:N]),forecast[:S])
    ρ̃ = zeros(length(gas_data[:N]),forecast[:S])
    exp_cost = zeros(forecast[:S])

    for s in 1:forecast[:S]
        φ̃[:,s] = sol_stochastic[:φ] .+ (lin_res[:ς̀2]*(sol_stochastic[:α] .- diagm(ones(length(gas_data[:N])))) .- lin_res[:ς̀3]*sol_stochastic[:β])*forecast[:ξ][:,s]
        ϑ̃[:,s] = sol_stochastic[:ϑ] .+ sol_stochastic[:α]*forecast[:ξ][:,s]
        κ̃[:,s] = sol_stochastic[:κ] .+ sol_stochastic[:β]*forecast[:ξ][:,s]
        π̃[:,s] = sol_stochastic[:𝛑] .+ (lin_res[:ς̆2]*(sol_stochastic[:α] .- lin_res[:ς̂3]*sol_stochastic[:β] .- diagm(ones(length(gas_data[:N])))))*forecast[:ξ][:,s]
        ρ̃[:,s] = sqrt.(max.(π̃[:,s],0))
        exp_cost[s] = ϑ̃[:,s]'*diagm(vec(gas_data[:c]))*ϑ̃[:,s]
    end

    inf_flag = zeros(forecast[:S])
    num_tolerance = 0.001
    for s in 1:forecast[:S]
        for n in gas_data[:N]
            if n in findall(x->x>0, gas_data[:ϑ̅])
                ϑ̃[n,s] >= gas_data[:ϑ̅][n] + num_tolerance ? inf_flag[s] = 1 : NaN
                ϑ̃[n,s] <= 0 - num_tolerance ? inf_flag[s] = 1 : NaN
            end
            ρ̃[n,s] >= gas_data[:ρ̅][n] + num_tolerance ? inf_flag[s] = 1 : NaN
            ρ̃[n,s] <= gas_data[:ρ̲][n] - num_tolerance ? inf_flag[s] = 1 : NaN
        end
        for p in gas_data[:E_a]
            φ̃[p,s] <= 0  - num_tolerance ? inf_flag[s] = 1 : NaN
            κ̃[p,s] >= gas_data[:κ̅][p]  + num_tolerance ? inf_flag[s] = 1 : NaN
            κ̃[p,s] <= gas_data[:κ̲][p]  - num_tolerance ? inf_flag[s] = 1 : NaN
        end
    end
    @info("empirical violation probability ---> $(sum(inf_flag)/forecast[:S]*100)%")
    return Dict(:φ => φ̃, :ϑ => ϑ̃, :κ => κ̃, :𝛑 => π̃, :ρ => ρ̃,
                :cost => mean(exp_cost), :ε_stat => sum(inf_flag)/forecast[:S])
end

function stochastic_dual_solution(gas_data,sol_stochastic,lin_res,forecast,settings)
    """
    dual problem of the chance-constrained gas network optimization
    """
    # extract dual variables
    λ_c = sol_stochastic[:λ_c]
    λ_w = sol_stochastic[:λ_w]
    λ_π̇ = sol_stochastic[:λ_π]
    λ_r = sol_stochastic[:λ_r]
    λ_ϑ = [sol_stochastic[:λ_μ_u_ϑ][i][1] for i in gas_data[:N]]
    λ_α = [sol_stochastic[:λ_μ_u_α][i][1] for i in gas_data[:N]]
    λ_π̅ = [sol_stochastic[:λ_u_π̅][i][1] for i in gas_data[:N]]
    λ_π̲ = [sol_stochastic[:λ_u_π̲][i][1] for i in gas_data[:N]]
    λ_ϑ̅ = [sol_stochastic[:λ_u_ϑ̅][i][1] for i in gas_data[:N]]
    λ_ϑ̲ = [sol_stochastic[:λ_u_ϑ̲][i][1] for i in gas_data[:N]]
    λ_κ̅ = [sol_stochastic[:λ_u_κ̅][i][1] for i in gas_data[:E]]
    λ_κ̲ = [sol_stochastic[:λ_u_κ̲][i][1] for i in gas_data[:E]]
    λ_s_π = [sol_stochastic[:λ_u_s_π][i][1] for i in gas_data[:N]]
    λ_s_φ = [sol_stochastic[:λ_u_s_φ][i][1] for i in gas_data[:E]]
    u_ϑ = [sol_stochastic[:λ_μ_u_ϑ][i][3] for i in gas_data[:N]]
    u_α = [sol_stochastic[:λ_μ_u_α][i][j] for i in gas_data[:N], j in 3:length(gas_data[:N])+2]
    u_π̅ = [sol_stochastic[:λ_u_π̅][i][j] for i in gas_data[:N], j in 2:length(gas_data[:N])+1]
    u_π̲ = [sol_stochastic[:λ_u_π̲][i][j] for i in gas_data[:N], j in 2:length(gas_data[:N])+1]
    u_ϑ̅ = [sol_stochastic[:λ_u_ϑ̅][i][j] for i in gas_data[:N], j in 2:length(gas_data[:N])+1]
    u_ϑ̲ = [sol_stochastic[:λ_u_ϑ̲][i][j] for i in gas_data[:N], j in 2:length(gas_data[:N])+1]
    u_κ̅ = [sol_stochastic[:λ_u_κ̅][i][j] for i in gas_data[:E], j in 2:length(gas_data[:N])+1]
    u_κ̲ = [sol_stochastic[:λ_u_κ̲][i][j] for i in gas_data[:E], j in 2:length(gas_data[:N])+1]
    u_s_π = [sol_stochastic[:λ_u_s_π][i][j] for i in gas_data[:N], j in 2:length(gas_data[:N])+1]
    u_s_φ = [sol_stochastic[:λ_u_s_φ][i][j] for i in gas_data[:E], j in 2:length(gas_data[:N])+1]
    λ_φ̲ = zeros(length(gas_data[:E]))
    u_φ̲ = zeros(length(gas_data[:E]),length(gas_data[:N]))
    for i in gas_data[:E]
        i ∈ gas_data[:E_a] ? λ_φ̲[i] = sol_stochastic[:λ_u_φ̲][i][1] : NaN
        i ∈ gas_data[:E_a] ? u_φ̲[i,:] = sol_stochastic[:λ_u_φ̲][i][2:end] : NaN
    end
    μ_ϑ = [sol_stochastic[:λ_μ_u_ϑ][i][2] for i in gas_data[:N]]
    μ_α =  [sol_stochastic[:λ_μ_u_α][i][2] for i in gas_data[:N]]

    # partial lagrangian function
    L_part = (
    + λ_c'*(gas_data[:A]*sol_stochastic[:φ] .- sol_stochastic[:ϑ] + gas_data[:B]*sol_stochastic[:κ] + gas_data[:δ])
    + λ_r'*(ones(length(gas_data[:N])) - (sol_stochastic[:α]-gas_data[:B]*sol_stochastic[:β])'*ones(length(gas_data[:N])))
    + λ_w'*(sol_stochastic[:φ] - lin_res[:ς1] - lin_res[:ς2]*sol_stochastic[:𝛑] - lin_res[:ς3]*sol_stochastic[:κ])
    - λ_s_φ'*sol_stochastic[:s_φ]
    - λ_s_π'*sol_stochastic[:s_𝛑]
    - λ_φ̲'*sol_stochastic[:φ]
    - λ_π̅'*(gas_data[:ρ̅].^2 - sol_stochastic[:𝛑])
    - λ_π̲'*(sol_stochastic[:𝛑] - gas_data[:ρ̲].^2)
    - sum((u_s_φ + Φ(settings,gas_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*(sol_stochastic[:α] - diagm(ones(length(gas_data[:N])))) - lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in gas_data[:E])
    - sum((u_s_π + Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*(sol_stochastic[:α] - lin_res[:ς̂3]*sol_stochastic[:β] - diagm(ones(length(gas_data[:N])))))[n,:] for n in gas_data[:N])
    )

    # partial lagrangian function decomposition
    R_free = + λ_w' * lin_res[:ς1]
    R_rent = (
            - λ_c'*gas_data[:A] * sol_stochastic[:φ]
            - λ_w' * sol_stochastic[:φ]
            + λ_w' * lin_res[:ς2] * sol_stochastic[:𝛑]
            + λ_φ̲'*sol_stochastic[:φ]
            + λ_π̅'*(gas_data[:ρ̅].^2 - sol_stochastic[:𝛑])
            + λ_π̲'*(sol_stochastic[:𝛑] - gas_data[:ρ̲].^2)
            + λ_s_φ'*sol_stochastic[:s_φ]
            + λ_s_π'*sol_stochastic[:s_𝛑]
            )
    R_inj = (
            + λ_c' * sol_stochastic[:ϑ]
            + λ_r' * sol_stochastic[:α]'*ones(length(gas_data[:N]))
            + sum((u_s_φ + Φ(settings,gas_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*sol_stochastic[:α])[l,:] for l in gas_data[:E])
            + sum((u_s_π + Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*sol_stochastic[:α])[n,:] for n in gas_data[:N])
            )
    R_act = (
            - λ_c' * gas_data[:B] * sol_stochastic[:κ]
            - λ_r' * (gas_data[:B]*sol_stochastic[:β])' * ones(length(gas_data[:N]))
            + λ_w' * lin_res[:ς3] * sol_stochastic[:κ]
            - sum((u_s_φ + Φ(settings,gas_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in gas_data[:E])
            - sum((u_s_π + Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*lin_res[:ς̂3]*sol_stochastic[:β])[n,:] for n in gas_data[:N])
            )
    R_con = (
            + λ_c' * gas_data[:δ]
            + λ_r' * ones(length(gas_data[:N]))
            + sum((u_s_φ + Φ(settings,gas_data)*u_φ̲)[l,:]'*forecast[:Σ½]*lin_res[:ς̀2][l,:] for l in gas_data[:E])
            + sum((u_s_π + Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*lin_res[:ς̆2][n,:] for n in gas_data[:N])
            )

    # revenue balance
    R_con - R_inj - R_act - R_rent - R_free
    # revenue decomposition
    R_inj_nom_bal = λ_c' * sol_stochastic[:ϑ]
    R_inj_rec_bal = λ_r' * sol_stochastic[:α]'*ones(length(gas_data[:N]))
    R_inj_net_lim = sum(Φ(settings,gas_data)*u_φ̲[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*sol_stochastic[:α])[l,:] for l in gas_data[:E]) + sum((Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*sol_stochastic[:α])[n,:] for n in gas_data[:N])
    R_inj_net_var = sum(u_s_φ[l,:]'*forecast[:Σ½]*(lin_res[:ς̀2]*sol_stochastic[:α])[l,:] for l in gas_data[:E]) + sum(u_s_π[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*sol_stochastic[:α])[n,:] for n in gas_data[:N])

    R_act_nom_bal = λ_w' * lin_res[:ς3] * sol_stochastic[:κ] - λ_c' * gas_data[:B] * sol_stochastic[:κ]
    R_act_rec_bal = - λ_r' * (gas_data[:B]*sol_stochastic[:β])' * ones(length(gas_data[:N]))
    R_act_net_lim = - sum((Φ(settings,gas_data)*u_φ̲)[l,:]'*forecast[:Σ½]*(lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in gas_data[:E]) - sum((Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*lin_res[:ς̂3]*sol_stochastic[:β])[n,:] for n in gas_data[:N])
    R_act_net_var = - sum(u_s_φ[l,:]'*forecast[:Σ½]*(lin_res[:ς̀3]*sol_stochastic[:β])[l,:] for l in gas_data[:E]) - sum(u_s_π[n,:]' * forecast[:Σ½]*(lin_res[:ς̆2]*lin_res[:ς̂3]*sol_stochastic[:β])[n,:] for n in gas_data[:N])

    R_con_nom_bal = λ_c' * gas_data[:δ]
    R_con_rec_bal = λ_r' * ones(length(gas_data[:N]))
    R_con_net_lim = sum((Φ(settings,gas_data)*u_φ̲)[l,:]'*forecast[:Σ½]*lin_res[:ς̀2][l,:] for l in gas_data[:E]) + sum((Φ(settings,gas_data)*u_π̅ + Φ(settings,gas_data)*u_π̲)[n,:]' * forecast[:Σ½]*lin_res[:ς̆2][n,:] for n in gas_data[:N])
    R_con_net_var = sum(u_s_φ[l,:]'*forecast[:Σ½]*lin_res[:ς̀2][l,:] for l in gas_data[:E]) + sum(u_s_π[n,:]' * forecast[:Σ½]*lin_res[:ς̆2][n,:] for n in gas_data[:N])

    Revenue_decomposition = DataFrame(source = ["nom_bal","rec_bal","net_lim","net_var","total"],
                                            inj = [R_inj_nom_bal,R_inj_rec_bal,R_inj_net_lim,R_inj_net_var, R_inj_nom_bal+R_inj_rec_bal+R_inj_net_lim+R_inj_net_var],
                                            act = [R_act_nom_bal,R_act_rec_bal,R_act_net_lim,R_act_net_var, R_act_nom_bal+R_act_rec_bal+R_act_net_lim+R_act_net_var],
                                            con = [R_con_nom_bal,R_con_rec_bal,R_con_net_lim,R_con_net_var, R_con_nom_bal+R_con_rec_bal+R_con_net_lim+R_con_net_var]
    )

    # compute indvidual revenues & profits
    R_ind_inj = [λ_c[n]*sol_stochastic[:ϑ][n] + (λ_r' + lin_res[:ς̆2][:,n]'*(u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲)*forecast[:Σ½] + lin_res[:ς̀2][:,n]'*(u_s_φ .+ Φ(settings,gas_data) * u_φ̲)*forecast[:Σ½])*sol_stochastic[:α][n,:] for n in gas_data[:N]]
    Π_inj = [R_ind_inj[n] - sol_stochastic[:c_ϑ][n] - sol_stochastic[:c_α][n] for n in gas_data[:N]]
    R_ind_act = [ -λ_c'*gas_data[:B][:,l] * sol_stochastic[:κ][l] + lin_res[:ς3][:,l]' * λ_w * sol_stochastic[:κ][l] - ones(length(gas_data[:N]))' * gas_data[:B][:,l] * λ_r' * sol_stochastic[:β][l,:] - (lin_res[:ς̆2]*lin_res[:ς̂3])[:,l]' * (u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲) * forecast[:Σ½] * sol_stochastic[:β][l,:] - lin_res[:ς̀3][:,l]' * (u_s_φ .+ Φ(settings,gas_data) * u_φ̲) * forecast[:Σ½] * sol_stochastic[:β][l,:] for l in gas_data[:E_a]]
    R_ind_rent = (- λ_c'*gas_data[:A] * sol_stochastic[:φ] - λ_w' * sol_stochastic[:φ] + λ_w' * lin_res[:ς2] * sol_stochastic[:𝛑] + λ_φ̲'*sol_stochastic[:φ] + λ_π̅'*(gas_data[:ρ̅].^2 - sol_stochastic[:𝛑]) + λ_π̲'*(sol_stochastic[:𝛑] - gas_data[:ρ̲].^2) + λ_s_φ'*sol_stochastic[:s_φ] + λ_s_π'*sol_stochastic[:s_𝛑])
    # R_ind_con = [λ_c[n]*gas_data[:δ][n] + λ_r[n] + sum(lin_res[:ς̀2][l,n]*(u_s_φ .+ Φ(settings,gas_data) * u_φ̲)[l,:]'*forecast[:Σ½][n,:] for l in gas_data[:E]) + sum(lin_res[:ς̆2][k,n]*(u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲)[k,:]'*forecast[:Σ½][n,:] for k in gas_data[:N]) for n in gas_data[:N]]
    R_ind_con = [λ_c[n]*gas_data[:δ][n] + λ_r[n] + forecast[:Σ½][n,:]'*(u_s_φ .+ Φ(settings,gas_data) * u_φ̲)'*lin_res[:ς̀2][:,n] + forecast[:Σ½][n,:]'*(u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲)'*lin_res[:ς̆2][:,n] for n in gas_data[:N]]

    # check stationarity conditions
    ∂L_∂s_φ = settings[:ψ_φ] .- λ_s_φ
    ∂L_∂s_π = settings[:ψ_𝛑] .- λ_s_π
    ∂L_∂c_ϑ = 1 .- μ_ϑ
    ∂L_∂c_α = 1 .- μ_α
    ∂L_∂π = λ_π̅ .- λ_π̲ .- vec(λ_w'*lin_res[:ς2]) ; ∂L_∂π[gas_data[:ref]] = ∂L_∂π[gas_data[:ref]] - λ_π̇
    ∂L_∂φ = vec(λ_c'*gas_data[:A]) .+ λ_w .- λ_φ̲
    ∂L_∂ϑ = - u_ϑ .* gas_data[:c̀] .- λ_c .+ λ_ϑ̅ .- λ_ϑ̲
    ∂L_∂κ = vec(λ_c'*gas_data[:B]) .- vec(λ_w'*lin_res[:ς3]) .+ λ_κ̅ .- λ_κ̲
    ∂L_∂α = zeros(length(gas_data[:N]),length(gas_data[:N]))
    ∂L_∂β = zeros(length(gas_data[:E]),length(gas_data[:N]))
    for n in gas_data[:N]
        # ∂L_∂α[n,:] = 2*gas_data[:c][n]*forecast[:Σ]*sol_stochastic[:α][n,:] .- λ_r .- forecast[:Σ½] * (u_φ' * lin_res[:ς̀2][:,n] + u_π' * lin_res[:ς̆2][:,n] + Φ(settings,gas_data) * u_ϑ̅[n,:] + Φ(settings,gas_data) * u_ϑ̲[n,:])
        ∂L_∂α[n,:] = -forecast[:Σ½]*u_α[n,:]*gas_data[:c̀][n] .- λ_r .- forecast[:Σ½] * ((u_s_φ .+ Φ(settings,gas_data) * u_φ̲)' * lin_res[:ς̀2][:,n] + (u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲)' * lin_res[:ς̆2][:,n] + Φ(settings,gas_data) * u_ϑ̅[n,:] + Φ(settings,gas_data) * u_ϑ̲[n,:])
    end
    for l in gas_data[:E]
        ∂L_∂β[l,:] = ones(length(gas_data[:N]))'*gas_data[:B][:,l]*λ_r .+ forecast[:Σ½] * (u_s_φ .+ Φ(settings,gas_data) * u_φ̲)' * lin_res[:ς̀3][:,l] .+ forecast[:Σ½] * (u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲)' * (lin_res[:ς̆2]*lin_res[:ς̂3])[:,l] .- forecast[:Σ½] * Φ(settings,gas_data) * (u_κ̅ + u_κ̲)[l,:]
    end
    # total mismatch
    mismatch = (sum(abs.(∂L_∂c_ϑ)) + sum(abs.(∂L_∂c_α)) + sum(abs.(∂L_∂s_φ))
                + sum(abs.(∂L_∂s_π)) + sum(abs.(∂L_∂π)) + sum(abs.(∂L_∂φ))
                + sum(abs.(∂L_∂ϑ)) + sum(abs.(∂L_∂κ)) + sum(abs.(∂L_∂β)))
    mismatch <= 1e-2 ? @info("stationarity conditions hold; total mismatch : $(round(mismatch,digits=8))") : @warn("stationarity conditions do not hold;  mismatch : $(round(mismatch,digits=8))")

    # compute dual objective function
    dual_obj = (λ_c' * gas_data[:δ]
    + sum(λ_r)
    - λ_w' * lin_res[:ς1]
    - 0.5 * sum(λ_ϑ)
    - 0.5 * sum(λ_α)
    - λ_π̅' * gas_data[:ρ̅].^2
    + λ_π̲' * gas_data[:ρ̲].^2
    - λ_ϑ̅' * gas_data[:ϑ̅]
    + λ_ϑ̲' * gas_data[:ϑ̲]
    - λ_κ̅' * gas_data[:κ̅]
    + λ_κ̲' * gas_data[:κ̲]
    + sum((u_s_φ .+ Φ(settings,gas_data) * u_φ̲)[l,:]' * forecast[:Σ½] * (lin_res[:ς̀2] * diagm(ones(length(gas_data[:N]))))[l,:] for l in gas_data[:E])
    + sum((u_s_π .+ Φ(settings,gas_data) * u_π̅ .+ Φ(settings,gas_data)*  u_π̲)[n,:]' * forecast[:Σ½] * (lin_res[:ς̆2] * diagm(ones(length(gas_data[:N]))))[n,:] for n in gas_data[:N])
    + λ_π̇*lin_res[:𝛑̇][gas_data[:ref]])
    # check duality gap
    duality_gap = norm(dual_obj-sol_stochastic[:obj])/sol_stochastic[:obj]*100
    duality_gap <= 1e-3 ? @info("strong duality holds; duality gap : $(round(duality_gap,digits=3))%") : @info("strong duality does not hold; duality gap : $(round(duality_gap,digits=3))%")

    # return dual solution
    return Dict(:dual_obj => dual_obj, :R_inj => round.(R_inj), :Π_inj => round.(Π_inj), :R_act => round.(R_act), :R_con => round.(R_con), :R_rent => round.(R_rent), :Revenue_decomposition => Revenue_decomposition)
end

function projection_opt(gas_data,forecast,sol_ofs,s)
    """
    projection problem
    """
    N_ϑ = findall(x->x>0, gas_data[:ϑ̅])
    N_κ = gas_data[:E_a]
    # build model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    # variable declaration
    @variable(model, 𝛑[1:length(gas_data[:N])])
    @variable(model, φ[1:length(gas_data[:E])])
    @variable(model, κ[1:length(gas_data[:E])])
    @variable(model, ϑ[1:length(gas_data[:N])])
    # minimize gas production cost
    @objective(model, Min, (sol_ofs[:ϑ][N_ϑ,s] .- ϑ[N_ϑ])' * diagm(ones(length(N_ϑ))) * (sol_ofs[:ϑ][N_ϑ,s] .- ϑ[N_ϑ])
                            + (sol_ofs[:κ][N_κ,s] .- κ[N_κ])' * diagm(ones(length(N_κ))) * (sol_ofs[:κ][N_κ,s] .- κ[N_κ])
                )
    # gas variable limits
    @constraint(model, inj_lim_max[i=gas_data[:N]], ϑ[i] <= gas_data[:ϑ̅][i])
    @constraint(model, inj_lim_min[i=gas_data[:N]], ϑ[i] >= gas_data[:ϑ̲][i])
    @constraint(model, pre_lim_max[i=gas_data[:N]], 𝛑[i] <= gas_data[:ρ̅][i]^2)
    @constraint(model, pre_lim_min[i=gas_data[:N]], 𝛑[i] >= gas_data[:ρ̲][i]^2)
    @constraint(model, com_lim_max[i=gas_data[:E]], κ[i] <= gas_data[:κ̅][i])
    @constraint(model, com_lim_min[i=gas_data[:E]], κ[i] >= gas_data[:κ̲][i])
    # gas flow equations
    @constraint(model, w_eq, φ .== lin_res[:ς1] + lin_res[:ς2] * 𝛑 + lin_res[:ς3] * κ)
    @constraint(model, gas_bal, ϑ .- gas_data[:δ] .- forecast[:ξ][:,s] .- gas_data[:B]*κ .== gas_data[:A]*φ)
    @constraint(model, φ_pl[l=gas_data[:E_a]], φ[l] >= 0)
    @constraint(model, 𝛑[gas_data[:ref]] == lin_res[:𝛑̇][gas_data[:ref]])
    # solve model
    optimize!(model)
    @info("projection terminates with status: $(termination_status(model))")
    # return solution
    solution = Dict(
    :Δϑ => sum(norm(JuMP.value.(ϑ)[n] - sol_ofs[:ϑ][n,s]) for n in N_ϑ),
    :Δκ => sum(sqrt(norm(JuMP.value.(κ)[n] - sol_ofs[:κ][n,s])) for n in N_κ),
    )
end


function projection_analysis(gas_data,forecast,sol_ofs)
    """
    obtain projection statistics
    """
    S = min(100,forecast[:S])
    Δϑ = zeros(S)
    Δκ = zeros(S)
    for s in 1:S
        sol_proj = projection_opt(gas_data,forecast,sol_ofs,s)
        Δϑ = sol_proj[:Δϑ]
        Δκ = sol_proj[:Δκ]
    end
    return mean(Δϑ), mean(Δκ)
end


#--Uncomment the following block for a single run--#
# Experiment settings:
 # settings[:det] => true retrieves the deterministic policies,
 # settings[:ψ_𝛑] = settings[:ψ_φ] = 0 retrieves variance-agnostic policies,
 # settings[:ε] regulates the joint constraint violation probability, and
 # settings[:σ] regulates the standard deviation of the forecast errors.
 """
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
"""
