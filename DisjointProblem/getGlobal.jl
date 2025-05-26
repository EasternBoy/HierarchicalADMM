function get_GlobalVars(root::linknode)
    varName = String[]
    getVarName!(root,varName)
    
    dict = Dict{String, VariableRef}()
    opti = JuMP.Model(Optim.Optimizer)
    setVariables!(root, dict, opti)
    
    J = NonlinearExpr[]
    getGlobalCostFunction!(root, dict, J)
    J = sum(J)

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

    dict_results = Dict{String, Float64}() #dimension of all variables are one
    for (name, var) in dict
        dict_results[name] = value(var)
    end
    return  dict_results
end

function setVariables!(node::linknode, dict::Dict, opti)
    if node.children !== nothing
        for child in node.children
            setVariables!(child, dict, opti)
        end
    else
        dict[string("x",node.ID)] = @variable(opti, base_name = string("x",node.ID))
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

function getGlobalCostFunction!(node::linknode, dict::Dict, J::Vector{NonlinearExpr})
    varName = String[]
    getVarName!(node, varName)
    c = parse(Float64, node.ID)
    vec_vars = [dict[name] for name in varName]
    push!(J, costfunction(vec_vars, c))
    if node.children !== nothing
        for child in node.children
            getGlobalCostFunction!(child, dict, J)
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
        dict[node.ID] = [dict_res[string("x",node.ID)]]
    end
end

function get_err!(node::linknode, dict::Dict, err::Vector{Float64})
    if node.children !== nothing
        push!(err, norm(dict[node.ID] - vectp(node)))
        for child in node.children
            get_err!(child, dict, err)
        end
    else
        push!(err, norm(dict[node.ID] - vectp(node)))
    end
end