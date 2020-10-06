# Stochastic Control and Pricing for Natural Gas Networks
Repository containing case data and code for the paper "Stochastic Control and Pricing for Natural Gas Networks" by Vladimir Dvorkin, Anubhav Ratha, Pierre Pinson and Jalal Kazempour. You can find the paper here: [arxiv.com url here]

If you use this code or parts of it, please cite this paper.

## Instructions:
The numerical experiments presented in the paper are implemented in Julia using the JuMP package. The non-convex problems are solved using the Ipopt solver, while Mosek is used for the convex second-order cone programming problems. Please refer to JuMP documentation on how to set up these solvers. The experiments require Julia 1.4. Information on the packages needed along with their versions are provided in the `Project.toml` file. You can load the packages in a new Julia environment directly using the Julia package manager. Alternatively, you can run the following code:

```
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

## Reproducing experiment results:
The file `main.jl` contains the various models. Uncommenting the block of code at the end of the file can be used to perform a single run of the various models. This file can be used to explore the code and to study the results of various models under different experimental settings. The file `numerical_experiments.jl` reproduces the results reported in the Table I, Figure 1 and Figure 2 of the paper.
