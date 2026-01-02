function get_GlobalVarsProximal(root::linknode)
    global tt_vars
    global var_map
    
    x0   = ones(tt_vars)

    f = ProximalAlgorithms.AutoDifferentiable(
        x -> total_cost(x, root),
        AutoZygote()
    )

    g = ProximalOperators.IndBox(0., Inf)

    solver = ProximalAlgorithms.PANOC(maxit = 1000, tol = 1e-6, verbose = false)

    solution, iterations = solver(x0 = x0, f = f, g = g)

    dict_results = Dict{String, Union{Float64, Vector{Float64}}}() #dimension of all variables are one
    for (name, var) in var_map
        dict_results[name] = Float64[]
        for itv in var
            dict_results[name] = [dict_results[name]; solution[itv[1]:itv[2]]]
        end
    end
    return  dict_results, total_cost(solution, root)
end

function get_err!(node::linknode, dict::Dict, err::Vector{Float64})
    if node.children !== nothing
        push!(err, norm(dict[node.ID] - vect_prime(node)))
        for child in node.children
            get_err!(child, dict, err)
        end
    else
        push!(err, norm(dict[node.ID] - vect_prime(node)))
    end
end

function split_var(x::Vector{Float64}, ID::String)
    global var_map

    res = Float64[]

    segments = var_map[ID]

    n = length(segments)

    for i in 1:n
        res = [res; x[segments[i][1]:segments[i][2]]]
    end
    return res
end

function tt_cost_func!(x, node::linknode, opt_cost)

    xlocal    = split_var(x, node.ID)
    p         = node.cost_func.para
    opt_cost.value += node.cost_func.val(xlocal; para = p)

    if node.children !==  nothing
        for child in node.children
            tt_cost_func!(x, child, opt_cost)
        end
    end
end


mutable struct OptCost value end
# (opt::OptCost)() = opt.value

function total_cost(x, root::linknode)
    opt_cost = OptCost(0.) #reset after each tree transverse

    tt_cost_func!(x, root, opt_cost)

    return opt_cost.value
end