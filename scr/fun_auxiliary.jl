# auxiliary functions
ns(l) = Int(net_data[:n_s][l])          #retrieve sending node of a pipeline
nr(l) = Int(net_data[:n_r][l])          #retrieve receiving node of a pipeline
function Φ(settings,net_data)
    """
    compute the safety factor
    """
    settings[:det] == true ? res = 0 : NaN
    settings[:det] == false ? res = quantile(Normal(0,1), 1 - settings[:ε]/net_data[:num_con]) : NaN
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
