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
