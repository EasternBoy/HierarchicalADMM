include("TreeTopology.jl")


opti = JuMP.Model(Optim.Optimizer)


@variable(opti, x21[1:2])
@variable(opti, x22[1:2])
@variable(opti, x12[1:2])
@variable(opti, x31[1:2])
@variable(opti, x32[1:2])
@variable(opti, x33[1:2])
@variable(opti, x24[1:2])


J = dot(x21 .- tree[2][1].cost, x21 .- tree[2][1].cost) + #Leaf nodes
    dot(x22 .- tree[2][2].cost, x22 .- tree[2][2].cost) + 
    dot(x12 .- tree[1][2].cost, x12 .- tree[1][2].cost) +
    dot(x31 .- tree[3][1].cost, x31 .- tree[3][1].cost) +
    dot(x32 .- tree[3][2].cost, x32 .- tree[3][2].cost) +
    dot(x33 .- tree[3][3].cost, x33 .- tree[3][3].cost) + 
    dot(x24 .- tree[2][4].cost, x24 .- tree[2][4].cost) + 
    dot([x21;x22] .- tree[1][1].cost, [x21;x22] .- tree[1][1].cost) + 
    dot([x31;x32;x33] .- tree[2][3].cost, [x31;x32;x33] .- tree[2][3].cost)+
    dot([x31;x32;x33;x24] .- tree[1][3].cost, [x31;x32;x33;x24] .- tree[1][3].cost) + 
    dot([x21;x22;x12;x31;x32;x33;x24] .- rootnode.cost, [x21;x22;x12;x31;x32;x33;x24] .- rootnode.cost) #root

    @objective(opti, Min, J)
    JuMP.optimize!(opti)

println(JuMP.value.(x21) - tree[2][1].couple_state)
println(JuMP.value.(x22) - tree[2][2].couple_state)
println(JuMP.value.(x12) - tree[1][2].couple_state)
println(JuMP.value.(x31) - tree[3][1].couple_state)
println(JuMP.value.(x32) - tree[3][2].couple_state)
println(JuMP.value.(x33) - tree[3][3].couple_state)
println(JuMP.value.(x24) - tree[2][4].couple_state)

