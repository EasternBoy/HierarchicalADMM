function get_CenVars(root::linknode)
    
    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    @variable(opti, x[1:root.nV])
    
    J = NonlinearExpr[]
    getCostFunction!(root, x, J, opti)
    J = sum(J)

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

    return  JuMP.value.(x), JuMP.value.(J)
end

function getCostFunction!(node::linknode, x, J::Vector{NonlinearExpr}, opti)

    para = node.cost_func.para
    func = node.cost_func.val
    w    = node.cost_func.w

    # L1 regularization in JuMP
    slack = @variable(opti, [1: node.nV])

    @constraint(opti,  slack .>=  x)
    @constraint(opti,  slack .>= -x)

    push!(J, func(x; para = para) + w*sum(slack))
    if node.children !== nothing
        for child in node.children
            getCostFunction!(child, x, J, opti)
        end
    end
end


function tt_cost_func!(node::linknode, opt_cost)

    xlocal          = node.prime
    p               = node.cost_func.para
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