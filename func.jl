using Base: Forward
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using DiffEqSensitivity
using Zygote
using Sundials
using ForwardDiff
using LinearAlgebra
using Random
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load

# ===============
# generate data
n_input = 3;
n_samples = 1000;
rng = MersenneTwister(0x7777777);
x = rand(rng, n_input, n_samples);
a = rand(rng, n_input);
b = rand(rng, n_input);
y = sum(sin.(a .* x) .+ b, dims=1) + 0.05*(2*rand(rng, 1, n_samples).-1)

# ===============
# setup networks
nn = Chain(
        Dense(n_input, 5),
        Dense(5, 1)
);
p_init, re = Flux.destructure(nn);
loss(p) = mse(re(p)(x), y);

# ===============
# train by adam
function train_SGD(p_init; opt=ADAM(0.01), n_epoch=1000)
    p_pred = deepcopy(p_init);
    losses = Vector{Float64}();
    for epoch in 1:n_epoch
        grad = Zygote.gradient(p->loss(p), p_pred)[1]
        update!(opt, p_pred, grad)
        push!(losses, loss(p_pred))
    end
    return p_pred, losses, (1:n_epoch)
end
function plot_loss(h, t, losses; line=(3,:solid), color=:blue, label="")
    plot!(h, t, losses, line=line, color=color, label=label, yscale=:log10)
    # plot(y', re(p_adam)(x)', line=:scatter, label="")
end

# ===============
# tain by ODE Solver
function dθdt!(dθ, θ, k, t)
    dθ .= -Zygote.gradient(p->loss(p), θ)[1]
end
# function Jac(θ, k, t)
#     return -ForwardDiff.jacobian(y -> Zygote.gradient(x -> loss(x), y)[1], θ)
# end
function train_ODE(p_init; solver=TRBDF2(), tend=1000)
    p_pred = deepcopy(p_init);
    losses = Vector{Float64}();
    prob = ODEProblem(dθdt!, p_pred, (0., tend))#; jac=Jac);
    sol = solve(prob, u0=p_pred, solver)#, atol=1e-15, maxiters=1000);
    # println(sol.destats)
    for p = sol.u
        push!(losses, loss(p))
    end
    p_pred = sol.u[end]
    return p_pred, losses, sol.t
end

# solve for ADAM opts
h = plot(size=(400,300))
pa = palette(:default)
for (i,lr) in enumerate([0.1, 0.03, 0.01, 0.003])
    @show lr
    @time p, losses, t = train_SGD(p_init; opt=ADAM(lr), n_epoch=floor(10/lr))
    plot_loss(h, t.*lr, losses; line=(2,:solid), color=pa[i],
                label="$(@sprintf("ADAM lr=%.0e", lr))")
end

solvers = [
    ["TRBDF2",       TRBDF2()      ],
    ["Rosenbrock23", Rosenbrock23()],
    ["Tsit5",        Tsit5()       ],
    ["KenCarp4",     KenCarp4()    ],
]
for (i,(name,solver)) in enumerate(solvers)
    @show name
    @time p, losses, t = train_ODE(p_init; solver=solver, tend=10.)
    plot_loss(h, t, losses; line=(2,:dash), color=pa[i],
                label="$(@sprintf("%s", name))")
end
xlabel!(h, "Time")
ylabel!(h, "Loss")
png(h, "compare")
display(h)
