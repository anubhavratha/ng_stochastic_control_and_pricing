function linearization(net_data,model)
    """
    extracts the senstivity coefficients from the non-convex Weymouth equation
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
    Jac_π = Jac[net_data[:E],net_data[:N]]
    Jac_φ = Jac[net_data[:E],length(net_data[:N]) .+ net_data[:E]]
    Jac_κ = Jac[net_data[:E],(length(net_data[:N]) .+ length(net_data[:E])) .+ net_data[:E]]
    # operating point
    𝛑̇ = opertaing_point[raw_index.(index.(var_nl))][net_data[:N]]
    φ̇ = opertaing_point[raw_index.(index.(var_nl))][length(net_data[:N]) .+ net_data[:E]]
    κ̇ = opertaing_point[raw_index.(index.(var_nl))][(length(net_data[:N]) .+ length(net_data[:E])) .+ net_data[:E]]
    # linearization coefficients
    ς1 = inv(Jac_φ) * (Jac_π * 𝛑̇ + Jac_κ * κ̇ + Jac_φ * φ̇)
    ς2 = -inv(Jac_φ) * Jac_π
    ς3 = -inv(Jac_φ) * Jac_κ
    # pressure-related
    ς̂2 = net_data[:A]*ς2
    ς̂3 = net_data[:B] + net_data[:A]*ς3
    ς̆2 = remove_col_and_row(ς̂2,net_data[:ref])
    ς̆2 = full_matrix(inv(ς̆2),net_data[:ref])
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
    sol_linearized = linearized_opt(net_data,lin_res)
    # check lienarization quality
    maximum(sol_non_convex[:𝛑] .- sol_linearized[:𝛑]) <= 1 ? @info("linearization successful") : @warn("linearization fails")
    return  lin_res
end
