using LinearAlgebra
using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators
using ProximalAlgorithms

quadratic_cost = ProximalAlgorithms.AutoDifferentiable(
    x -> dot([3.4 1.2; 1.2 4.5] * x, x) / 2 + dot([-2.3, 9.9], x),
    AutoZygote(),
)
box_indicator = ProximalOperators.IndBox(0, 1)

ffb = ProximalAlgorithms.FastForwardBackward(maxit = 1000, tol = 1e-5, verbose = true)

solution, iterations = ffb(x0 = ones(2), f = quadratic_cost, g = box_indicator)