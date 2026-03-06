using JuMP
import Clarabel
import LinearAlgebra
import MathOptInterface as MOI


A = [3 2 4; 2 0 2; 4 2 3]
I = Matrix{Float64}(LinearAlgebra.I, 3, 3)
model = Model(Clarabel.Optimizer)
set_silent(model)
@variable(model, t)
@objective(model, Min, t)
@constraint(model, t .* I - A in PSDCone())
optimize!(model)
assert_is_solved_and_feasible(model)
objective_value(model)