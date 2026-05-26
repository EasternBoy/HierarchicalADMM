#Vectorize prime dict of a node in the same order as its children
function ProximalAlgorithms.value_and_gradient(cost_func::CostFunc, x)
    prm = cost_func.para
    q = cost_func.q
    λ = cost_func.λ

    val = cost_func.val(x; para = prm) + 1 / (2λ) * dot(x - q, x - q)
    grad = cost_func.grad(x; para = prm) + 1 / λ * (x - q)
    return val, grad
end

function Proximal_Iteration(cost_func::CostFunc, q::Union{Float64, Vector{Float64}}, λ::Float64)
    cost_func.q = q
    cost_func.λ = λ

    f = cost_func
    g = ProximalOperators.NormL1(cost_func.w)
    return f, g
end

function constrained_proximal_iteration(cost_func::CostFunc, q::Vector{Float64}, λ::Float64, x0::Vector{Float64})
    para = cost_func.para
    func = cost_func.val
    w = cost_func.w
    n = length(q)

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x = @variable(opti, [i = 1:n], start = x0[i])
    slack = @variable(opti, [i = 1:n])

    @constraint(opti, slack .>= x)
    @constraint(opti, slack .>= -x)
    @constraint(opti, local_l <= sum(x))
    @constraint(opti, sum(x) <= local_u)

    @objective(opti, Min, func(x; para = para) + w * sum(slack) + 1 / (2λ) * sum((x[i] - q[i])^2 for i in 1:n))

    JuMP.optimize!(opti)

    return JuMP.value.(x), 0
end

function edge_residual(parent::linknode, child::linknode)
    return parent.prime[child.ID] .- sum(vect_prime(child))
end

function constrained_aggregation_iteration(node::linknode; λ = λh)
    para = node.cost_func.para
    func = node.cost_func.val
    w = node.cost_func.w
    n = node.nV

    opti = JuMP.Model(Ipopt.Optimizer)
    set_silent(opti)

    x0 = vect_prime(node)
    x = @variable(opti, [i = 1:n], start = x0[i])
    slack = @variable(opti, [i = 1:n])

    @constraint(opti, slack .>= x)
    @constraint(opti, slack .>= -x)
    @constraint(opti, local_l <= sum(x))
    @constraint(opti, sum(x) <= local_u)

    penalty = zero(AffExpr)

    if node.parent !== nothing
        top_query = node.parent.prime[node.ID][1] + node.parent.dual[node.ID][1]
        penalty += (sum(x) - top_query)^2
        com_cost!(node.parent, [top_query], 1)
    end

    if node.children !== nothing
        for (idx, child) in enumerate(node.children)
            child_query = sum(vect_prime(child)) - node.dual[child.ID][1]
            penalty += (x[idx] - child_query)^2
        end
    end

    @objective(opti, Min, func(x; para = para) + w * sum(slack) + 1 / (2λ) * penalty)

    JuMP.optimize!(opti)

    return JuMP.value.(x), 0
end

function proxAlg!(node::linknode; λ = λh)
    solution, iterations = constrained_aggregation_iteration(node; λ = λ)

    if node.children === nothing
        node.prime[node.ID] = solution
    else
        for (idx, child) in enumerate(node.children)
            node.prime[child.ID] = [solution[idx]]
        end
    end
end

function hierarchicalADMM!(node::linknode, ter::Vector{Float64}; λ = λh)
    node.iteration += 1

    proxAlg!(node; λ = λ)

    if node.children !== nothing
        for child in node.children
            hierarchicalADMM!(child, ter; λ = λ)

            child_prime = vect_prime(child)
            res = edge_residual(node, child)
            node.dual[child.ID] += res

            com_cost!(child, child_prime, 1)
            push!(ter, maximum(abs.(res)))
        end
    end
end

function max_primal_residual(node::linknode)
    max_res = 0.0

    if node.children !== nothing
        for child in node.children
            res = edge_residual(node, child)
            max_res = max(max_res, maximum(abs.(res)))
            max_res = max(max_res, max_primal_residual(child))
        end
    end

    return max_res
end

function snapshot_edge_parent_primes!(node::linknode, snapshot::Dict)
    if node.children !== nothing
        for child in node.children
            snapshot[(node.ID, child.ID)] = copy(node.prime[child.ID])
            snapshot_edge_parent_primes!(child, snapshot)
        end
    end
end

function max_dual_residual(node::linknode, prev_parent_primes::Dict)
    max_res = 0.0

    if node.children !== nothing
        for child in node.children
            edge_key = (node.ID, child.ID)
            res = node.prime[child.ID] - prev_parent_primes[edge_key]
            max_res = max(max_res, maximum(abs.(res)))
            max_res = max(max_res, max_dual_residual(child, prev_parent_primes))
        end
    end

    return max_res
end

function hADMM(root::linknode, dict_result::Dict; tol = tol, λ = λh, max_iter = max_iter, return_residuals = false)
    global stop_arr

    traj_err = Float64[]
    traj_res = Float64[]
    traj_primal_res = Float64[]
    traj_dual_res = Float64[]
    opt_value = Float64[]
    traj_com = Float64[]
    traj_root_com = Float64[]

    initial_err = Float64[]
    get_err!(root, dict_result, initial_err)
    push!(traj_err, sum(initial_err))
    push!(traj_res, NaN)
    push!(traj_primal_res, NaN)
    push!(traj_dual_res, NaN)
    push!(opt_value, total_cost(root))
    push!(traj_com, 0.0)
    push!(traj_root_com, 0.0)

    for iteration in 1:max_iter
        ter = Float64[]
        prev_parent_primes = Dict()
        snapshot_edge_parent_primes!(root, prev_parent_primes)

        hierarchicalADMM!(root, ter; λ = λ)

        err = Float64[]
        get_err!(root, dict_result, err)
        push!(opt_value, total_cost(root))
        push!(traj_res, maximum(ter))
        push!(traj_primal_res, max_primal_residual(root))
        push!(traj_dual_res, max_dual_residual(root, prev_parent_primes))
        push!(traj_err, sum(err))

        total, _ = tt_com_iter(root)
        push!(traj_com, total["com"])
        push!(traj_root_com, root.com_cost)

        if maximum(ter) < tol
            println("hADMM converged after $iteration iterations in root")
            if return_residuals
                return traj_err, traj_res, opt_value, traj_com, traj_root_com, traj_primal_res, traj_dual_res
            end
            return traj_err, traj_res, opt_value, traj_com, traj_root_com
        end
    end

    if return_residuals
        return traj_err, traj_res, opt_value, traj_com, traj_root_com, traj_primal_res, traj_dual_res
    end
    return traj_err, traj_res, opt_value, traj_com, traj_root_com
end
