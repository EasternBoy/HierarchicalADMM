function get_CenVars(root::linknode)
    varName = String[]
    getVarName!(root,varName)
    
    dict = Dict()
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)
    setVariables!(root, dict, opti)
    
    J = NonlinearExpr[]
    getCostFunction!(root, dict, J, opti)
    J = sum(J)

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

    dict_vars = Dict() #dimension of all variables are one
    for (name, var) in dict
        dict_vars[name] = JuMP.value.(var)
    end

    dict_result = Dict()
    assign_result!(root, dict_result, dict_vars)

    return  dict_result, JuMP.value.(J)
end

function setVariables!(node::linknode, dict::Dict, opti)
    if node.children !== nothing
        for child in node.children
            setVariables!(child, dict, opti)
        end
    else
        dict[string("x",node.ID)] = @variable(opti, [i=1:node.nV], base_name = string("x",node.ID))
        # dict[string("x",node.ID)] = @variable(opti, base_name = string("x",node.ID))
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

function getCostFunction!(node::linknode, dict::Dict, J::Vector{NonlinearExpr}, opti)
    varName = String[]
    getVarName!(node, varName)
    # vec_vars = [dict[name] for name in varName]
    vec_vars = []
    for name in varName append!(vec_vars, dict[name]) end 

    para = node.cost_func.para
    func = node.cost_func.val
    w    = node.cost_func.w

    # L1 regularization in JuMP
    n    = length(vec_vars)
    slack = @variable(opti, [1:n])

    @constraint(opti,  slack .>=  vec_vars)
    @constraint(opti,  slack .>= -vec_vars)

    push!(J, func(vec_vars; para = para) + w*sum(slack))
    if node.children !== nothing
        for child in node.children
            getCostFunction!(child, dict, J, opti)
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
    push!(err, norm(dict[node.ID] - vect_prime(node)))
    if node.children !== nothing
        for child in node.children
            get_err!(child, dict, err)
        end
    end
end


function tt_cost_func!(node::linknode, opt_cost)

    xlocal         = vect_prime(node)
    p              = node.cost_func.para
    opt_cost.value += node.cost_func.val(xlocal; para = p) + node.cost_func.w * norm(xlocal, 1)

    if node.children !==  nothing
        for child in node.children
            tt_cost_func!(child, opt_cost)
        end
    end
end

mutable struct OptCost value end

function total_cost(root::linknode)
    opt_cost = OptCost(0.)

    tt_cost_func!(root, opt_cost)

    return opt_cost.value
end