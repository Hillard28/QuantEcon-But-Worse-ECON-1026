#==============================================================================
# ECON-GA 1026 Problem Set 3
# Ryan Gilland
==============================================================================#
import LinearAlgebra as linalg
import Statistics as stats
import Distributions as dist
import QuantEcon as qe
import Random as random
import Plots as plt

random.seed!(1383686510)

#==============================================================================
# Utility functions
==============================================================================#
function plot_series(y; ylim, display=true)
    x = range(0, length(y)-1, length=length(y))
    series_plot = plt.plot(x, y; ylims=(-ylim, ylim))
    if display
        plt.display(series_plot)
    end
end

function plot_histogram(y; bins=:auto, display=true)
    histogram_plot = plt.histogram(y, bins=bins)
    if display
        plt.display(histogram_plot)
    end
end

function print_theta(theta, N)
    for i = 1:N
        println(round.(theta[(i-1)*N+1:i*N], digits=3))
    end
end

#==============================================================================
# Compute Moments
==============================================================================#
vₑ = 0.06
σₑ = sqrt(vₑ)
μₑ = 0
Dₑ = dist.Normal(μₑ, σₑ)

ρ = 0.90
w̄ = -vₑ / (2*(1 + ρ))

tmean = w̄ / (1 - ρ)
tvar = vₑ / (1 - ρ^2)

# For when we adjust ρ
vₑ98 = (1 - 0.98^2) * tvar
σₑ98 = sqrt(vₑ98)
Dₑ98 = dist.Normal(μₑ, σₑ98)
w̄98 = -vₑ98 / (2*(1 + 0.98))
tmean98 = w̄98 / (1 - 0.98)
tvar98 = vₑ98 / (1 - 0.98^2)

#==============================================================================
# Tauchen / Rouwenhorst
==============================================================================#
# Configure y (Tauchen)
"""
    tauchen(
    mean,
    variance,
    N,
    m,
    rho;
    print_output=false
    )

Discretizes and simulates a continuous AR(1) process

Positional Args
mean: mean of the process being discretized
variance: variance of the random process
N: number of possible realizations yₖ
m: multiple applied to unconditional standard deviation
rho: coefficient on lag term

Keyword Args
print_output: debugging tool, prints some variables
"""
function tauchen(
    mean,
    variance,
    N,
    m,
    rho;
    print_output=false
    )
    σ = sqrt(variance)
    v = variance
    μ = mean
    states = Array{Float64}(undef, N)
    states[N] = m*sqrt(v / (1 - rho^2))
    states[1] = -states[N]
    if print_output
        println("States:")
        println(states[1])
    end
    d = (states[N] - states[1])/(N - 1)
    for i = 2:N-1
        states[i] = states[1] + (i - 1)*d
        if print_output
            println(states[i])
        end
    end
    if print_output
        println(states[N])
    end
    
    # Standard normal distribution for the normalized states
    D = dist.Normal(0, 1)
    
    # Transition matrix
    Θ = Array{Float64}(undef, N^2)
    for i = 1:N
        for j = 1:N
            if j == 1
                Θ[(i-1)*N + j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/σ)
            elseif j == N
                Θ[(i-1)*N + j] = 1 - dist.cdf(D, (states[j] - d/2 - rho*states[i])/σ)
            else
                Θ[(i-1)*N + j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/σ) -
                    dist.cdf(D, (states[j] - d/2 - rho*states[i])/σ)
            end
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[(i-1)*N+1:i*N], digits=3))
        end
    end
    
    states .+= μ / (1 - rho)

    return Θ, states
end

"""
    rouwenhorst(
    mean,
    variance,
    N,
    rho;
    print_output=false
    )

Discretizes and simulates a continuous AR(1) process

Positional Args
mean: mean of the process being discretized
variance: variance of the random process
N: number of possible realizations yₖ
rho: coefficient on lag term

Keyword Args
print_output: debugging tool, prints some variables
"""
function rouwenhorst(
    mean,
    variance,
    N,
    rho;
    print_output=false
    )
    v = variance
    μ = mean

    ψ = sqrt(v / (1 - rho^2)) * sqrt(N - 1)

    states = Array{Float64}(undef, N)
    states[N] = ψ
    states[1] = -ψ
    if print_output
        println("States:")
        println(states[1])
    end
    d = (states[N] - states[1])/(N - 1)
    for i = 2:N-1
        states[i] = states[1] + (i - 1)*d
        if print_output
            println(states[i])
        end
    end
    if print_output
        println(states[N])
    end

    p = (1 + rho) / 2
    q = p
    
    # Transition matrix
    Θₙ = Array{Float64}(undef, 4)
    Θₙ[1] = p
    Θₙ[2] = 1 - p
    Θₙ[3] = 1 - q
    Θₙ[4] = q
    Θ = Θₙ
    if N > 2
        global Θ
        global Θₙ
        for n = 3:N
            global Θ
            global Θₙ
            Θ = zeros(n^2)
            for i = 1:n
                for j = 1:n
                    if i <= n-1 && j <= n-1
                        Θ[(i-1)*n + j] += p*Θₙ[(i-1)*(n-1) + j]
                    end
                    if i <= n-1 && j >= 2
                        Θ[(i-1)*n + j] += (1 - p)*Θₙ[(i-1)*(n-1) + (j-1)]
                    end
                    if i >= 2 && j <= n-1
                        Θ[(i-1)*n + j] += (1 - q)*Θₙ[(i-2)*(n-1) + j]
                    end
                    if i >= 2 && j >= 2
                        Θ[(i-1)*n + j] += q*Θₙ[(i-2)*(n-1) + (j-1)]
                    end
                end
            end
            for i = 2:n-1
                for j = 1:n
                    Θ[(i-1)*n + j] /= 2
                end
            end
            Θₙ = Θ
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[(i-1)*N+1:i*N], digits=3))
        end
    end

    states .+= μ / (1 - rho)
    
    return Θ, states
end

"""
    simulate(theta, states, periods=10000, burn=0, replications=0)

Simulates a Markov Chain

Positional Args
theta: transition probability matrix
states: possible realized states
periods: length of simulation
burn: sets the number of iterations to run prior to the simulation
replications: number of simulations to run (unimplemented)
"""
function simulate(theta, states, periods=1000, burn=0, replications=0)
    N = length(states)
    
    # Configure CDF
    Θ = Array{dist.Categorical{Float64, Vector{Float64}}}(undef, N)
    for i = 1:N
        Θ[i] = dist.Categorical(theta[(i-1)*N+1:i*N])
    end

    # Initialize index of realized states and run burn-in
    index_series = Array{Int64}(undef, periods)
    index_series[1] = rand(1:N)
    if burn > 0
        for _ = 1:burn
            index_series[1] = dist.rand(Θ[index_series[1]], 1)[1]
        end
    end

    # Run simulation
    for t = 2:periods
        index_series[t] = dist.rand(Θ[index_series[t-1]], 1)[1]
    end
    
    state_series = Array{Float64}(undef, periods)
    for t = 1:periods
        state_series[t] = states[index_series[t]]
    end

    #=
    if replications > 0
        TBD
    end
    =#

    return state_series
end

#==============================================================================
# Problem 1 Results
==============================================================================#
Θ, states = tauchen(w̄, vₑ, 5, 3, 0.9; print_output=false)
yts5 = simulate(Θ, states, 10000, 5, 0)
Θ, states = tauchen(w̄, vₑ, 11, 3, 0.9; print_output=false)
yts11 = simulate(Θ, states, 10000, 15, 0)
Θ, states = tauchen(w̄98, vₑ98, 5, 3, 0.98; print_output=false)
yts5p = simulate(Θ, states, 10000, 5, 0)
Θ, states = rouwenhorst(w̄, vₑ, 5, 0.9; print_output=false)
yrs5 = simulate(Θ, states, 10000, 5, 0)
Θ, states = rouwenhorst(w̄98, vₑ98, 5, 0.98; print_output=false)
yrs5p = simulate(Θ, states, 10000, 5, 0)

println("\n\nProblem 1a)")
println("Sample moments (T5):")
println("Mean: ", round(stats.mean(yts5), digits=3))
println("Variance: ", round(stats.var(yts5), digits=3))

println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\n\nProblem 1b)")
println("Sample moments (T11):")
println("Mean: ", round(stats.mean(yts11), digits=3))
println("Variance: ", round(stats.var(yts11), digits=3))

println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\n\nProblem 1c)")
println("Sample moments (R5):")
println("Mean: ", round(stats.mean(yrs5), digits=3))
println("Variance: ", round(stats.var(yrs5), digits=3))

println("\nSample moments (T5):")
println("Mean: ", round(stats.mean(yts5), digits=3))
println("Variance: ", round(stats.var(yts5), digits=3))

println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\n\nProblem 1d)")
println("Sample moments (R5, ρ=0.98):")
println("Mean: ", round(stats.mean(yrs5p), digits=3))
println("Variance: ", round(stats.var(yrs5p), digits=3))

println("\nSample moments (T5, ρ=0.98):")
println("Mean: ", round(stats.mean(yts5p), digits=3))
println("Variance: ", round(stats.var(yts5p), digits=3))

println("\nTheoretical moments (ρ=0.98):")
println("Mean: ", round(tmean98, digits=3))
println("Variance: ", round(tvar98, digits=3))

#==============================================================================
# Sanity Check
==============================================================================#
mc5qe = qe.tauchen(5, ρ, σₑ, w̄, 3)
yts5qe = qe.simulate(mc5qe, 100000)
mc11qe = qe.tauchen(11, ρ, σₑ, w̄, 3)
yts11qe = qe.simulate(mc11qe, 100000)

mcr5qe = qe.rouwenhorst(5, ρ, σₑ, w̄)
yrs5qe = qe.simulate(mcr5qe, 100000)
mcr598qe = qe.rouwenhorst(5, 0.98, σₑ98, w̄98)
yrs598qe = qe.simulate(mcr598qe, 100000)

println("\nProblem 1a) (QuantEcon)")
println("Sample moments:")
println("Mean: ", round(stats.mean(yts5qe), digits=3))
println("Variance: ", round(stats.var(yts5qe), digits=3))

println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\nProblem 1b) (QuantEcon)")
println("Sample moments:")
println("Mean: ", round(stats.mean(yts11qe), digits=3))
println("Variance: ", round(stats.var(yts11qe), digits=3))

println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\nProblem 1c) (QuantEcon)")
println("Sample moments:")
println("Mean: ", round(stats.mean(yrs5qe), digits=3))
println("Variance: ", round(stats.var(yrs5qe), digits=3))

println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\nProblem 1d) (QuantEcon)")
println("Sample moments:")
println("Mean: ", round(stats.mean(yrs598qe), digits=3))
println("Variance: ", round(stats.var(yrs598qe), digits=3))

println("\nTheoretical moments (ρ=0.98):")
println("Mean: ", round(tmean98, digits=3))
println("Variance: ", round(tvar98, digits=3))
