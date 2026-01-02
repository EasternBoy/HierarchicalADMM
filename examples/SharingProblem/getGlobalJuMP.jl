function get_GlobalVarsJuMP(root::linknode)
    varName = String[]
    getVarName!(root,varName)
    
    dict = Dict{String, Union{VariableRef, Vector{VariableRef}}}()
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    setVariables!(root, dict, opti)
    setConstraints!(root, dict, opti)
    
    J = NonlinearExpr[]
    getGlobalCostFunction!(root, dict, J, opti)
    J = sum(J)

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

    dict_results = Dict{String, Union{Float64, Vector{Float64}}}() #dimension of all variables are one
    for (name, var) in dict
        dict_results[name] = JuMP.value.(var)
    end
    return  dict_results, JuMP.value.(J)
end

function setVariables!(node::linknode, dict::Dict, opti)
    if node.children === nothing
        x0 = ones(node.nV)
        dict[string("x",node.ID)] = @variable(opti, [i=1:node.nV], base_name = string("x",node.ID), start = x0[i])
    else
        for child in node.children
            setVariables!(child, dict, opti)
        end
    end
end

function setConstraints!(node::linknode, dict::Dict, opti)
    if node.children === nothing
        @constraint(opti, dict[string("x",node.ID)] .>= 0) #Constraints only are in leaf nodes
    else
        for child in node.children
            setConstraints!(child, dict, opti)
        end
    end
end

function getVarName!(node::linknode, str::Vector{String})
    if node.children !== nothing
        for child in node.children
            getVarName!(child, str)
        end
    else
        push!(str,string("x",node.ID))
    end
end

function getGlobalCostFunction!(node::linknode, dict::Dict, J::Vector{NonlinearExpr}, opti)
    varName = String[]
    getVarName!(node, varName)
    vec_vars = VariableRef[]

    for name in varName
        vec_vars = [vec_vars; dict[name]]
    end

    f = node.cost_func.val
    p = node.cost_func.para

    if node.children !== nothing  && node.parent !== nothing
        η  = p[1]
        τ  = p[2]
        nc = p[3]
        horizon = Int64(length(vec_vars)/nc)

        t = @variable(opti, [i=1:horizon], base_name = string("t",node.ID))
        @constraint(opti, t .>= 0)
        for i in 1:horizon
            @constraint(opti, t[i] >= sum([vec_vars[(j-1)*horizon + i] for j in 1:nc]) - τ)
        end
        push!(J, η*sum(t))
    else
        push!(J, f(vec_vars; para = p))
    end



    if node.children !== nothing
        for child in node.children
            getGlobalCostFunction!(child, dict, J, opti)
        end
    end
end


function assign_result!(node::linknode, dict::Dict, dict_res::Dict)
    if node.children !== nothing
        dict[node.ID] = Float64[]
        for child in node.children
            assign_result!(child, dict, dict_res::Dict)
            dict[node.ID] = [dict[node.ID]; dict[child.ID]]
        end
    else
        dict[node.ID] = dict_res[string("x",node.ID)]
    end
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

function tt_cost_func!(x, node::linknode)
    global opt_cost

    xlocal    = split_var(x, node.ID)
    p         = node.cost_func.para
    opt_cost += node.cost_func.val(xlocal; para = p)

    if node.children !==  nothing
        for child in node.children
            tt_cost_func!(x, child)
        end
    end
end

function total_cost(x, root::linknode)
    global opt_cost = 0. #reset after each tree transverse

    tt_cost_func!(x, root)

    return opt_cost
end