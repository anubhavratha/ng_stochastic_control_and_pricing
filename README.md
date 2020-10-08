# Stochastic Control and Pricing for Natural Gas Networks
This repository contains data and codes for the paper __Stochastic Control and Pricing for Natural Gas Networks__ by Vladimir Dvorkin, Anubhav Ratha, Pierre Pinson, and Jalal Kazempour. The paper is available on [arXiv](https://arxiv.org/abs/2010.03283). If you use this code or parts of it, please cite the paper.

<img width="875" alt="Screenshot 2020-10-08 at 10 30 54" src="https://user-images.githubusercontent.com/31773955/95434288-5cde3700-0951-11eb-8f01-93e0028da668.png">

## Installation
The optimization models were implemented in [Julia](https://juliacomputing.com/products/juliapro) (v.1.4) using [JuMP](https://github.com/JuliaOpt/JuMP.jl) modeling language for mathematical optimization embedded in Julia. 

The non-convex models are solved using the [Ipopt](https://ipoptjl.readthedocs.io/en/latest/ipopt.html) solver, while the [Mosek](https://www.mosek.com) solver is used for the convex second-order cone programming models. The Mosek solver needs to be licensed (free of charge for academic use).  Please refer to JuMP documentation on how to set up these solvers. 

The packages used and their versions are provided in the `Project.toml` file. To activate the packages in ```Project.toml```, open a terminal, clone the project using ```git clone```, ```cd``` to the project directory and call
```
$ julia 
julia> ]
pkg> activate .
pkg> instantiate
```
where ```julia``` is an alias to Julia installation. To run the code, type
```
julia> include("main.jl")
```
By default, the program outputs: 

<img width="677" alt="Screenshot 2020-10-08 at 10 57 57" src="https://user-images.githubusercontent.com/31773955/95437348-286c7a00-0955-11eb-9e77-8d7745f8c09f.png">


## Reproducing results:
To reproduce the results, specify the experiment settings in ```exp_settings``` dictionary contained in ```main.jl```. The experiment settings are detailed in Section V of the paper. To run the projection analisys, set ```:proj => true``` in the dictionary, but expect more running time.






# Stochastic Control and Pricing for Natural Gas Networks
Repository containing case data and code for the paper "Stochastic Control and Pricing for Natural Gas Networks" by Vladimir Dvorkin, Anubhav Ratha, Pierre Pinson and Jalal Kazempour. 
You can find the paper here: https://arxiv.org/abs/2010.03283

If you use this code or parts of it, please cite the paper.

## Instructions:
The numerical experiments presented in the paper are implemented in Julia using the JuMP package. The non-convex problems are solved using the Ipopt solver, while Mosek is used for the convex second-order cone programming problems. Please refer to JuMP documentation on how to set up these solvers. The experiments require Julia 1.4. Information on the packages needed along with their versions are provided in the `Project.toml` file. You can load the packages in a new Julia environment directly using the Julia package manager. Alternatively, you can run the following code:

```
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

## Reproducing experiment results:
The file `main.jl` contains the various models. Uncommenting the block of code at the end of the file can be used to perform a single run of the various models. This file can be used to explore the code and to study the results of various models under different experimental settings. The file `numerical_experiments.jl` reproduces the results reported in the Table I, Figure 1 and Figure 2 of the paper.
