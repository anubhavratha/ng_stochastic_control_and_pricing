function load_network_data(case)
    """
    extracts, transforms and loads network data
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
    net_data = Dict(:c => c, :c̀ => c̀, :ϑ̅ => ϑ̅, :ϑ̲ => ϑ̲, :δ => δ, :ρ̅ => ρ̅, :ρ̲ => ρ̲,
                    :n_s => n_s, :n_r => n_r, :k => k, :κ̅ => κ̅, :κ̲ => κ̲, :A => A, :B => B, :ref => ref,
                    :num_con => num_con, :E => E, :E_a => E_a, :N => N)
    @info("done loading network data")
    return net_data
end

function extract_forecast_data(net_data,settings)
    """
    extracts covariance matrix, its factorization and samples
    """
    N = length(net_data[:N])
    N_δ = findall(x->x!=0, net_data[:δ])
    N_δ̸ = setdiff(1:N,N_δ)
    I_δ = zeros(N); I_δ[N_δ] .= 1;

    Σ  = zeros(N,N)                     # extended covariance matrix
    Σ½ = zeros(N,N)                     # extended Cholesky factorization

    σ = zeros(N)                        # vector of standard deviations

    c = 0.00                            # correlation coefficient
    C = zeros(N,N)                      # correlation matrix

    for k in N_δ, j in N_δ
        σ[k] = settings[:σ]*net_data[:δ][k]
        k != j ? C[k,j] = c : NaN
        k == j ? C[k,j] = 1 : NaN
    end

    Σ = cor2cov(C, σ)
    F = cholesky(Σ[N_δ,N_δ]).L          # Cholesky factorization: F*F' = Σ[N_δ,N_δ]
    Σ½[N_δ,N_δ] = F

    S = settings[:S]
    ξ = zeros(N,S)
    ξ[N_δ,:] = rand(MvNormal(zeros(length(N_δ)),Σ[N_δ,N_δ]),S)

    return forecast = Dict(:Σ => Σ, :Σ½ => Σ½, :ξ => ξ, :σ => σ, :I_δ => I_δ, :N_δ => N_δ, :N_δ̸ => N_δ̸, :S => S)
end
