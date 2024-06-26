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

#==============================================================================
# Compute Moments
==============================================================================#
vₑ = 0.06
σₑ = sqrt(vₑ)
μₑ = 0
Dₑ = dist.Normal(μₑ, σₑ)
ρ = 0.90

# Set w̄ and moments according to analytical derivations
w̄ = -vₑ / (2*(1 + ρ))
tmean = w̄ / (1 - ρ)
tvar = vₑ / (1 - ρ^2)

# For when we adjust ρ
vₑ98 = (1 - 0.98^2) * tvar
σₑ98 = sqrt(vₑ98)
Dₑ98 = dist.Normal(μₑ, σₑ98)
ρ98 = 0.98
w̄98 = -vₑ98 / (2*(1 + 0.98))
tmean98 = w̄98 / (1 - 0.98)
tvar98 = vₑ98 / (1 - 0.98^2)

#==========================================================================================
# Markov Chain
==========================================================================================#
mutable struct MarkovChain
    theta::Array{Float64}
    states::Array{Float64}
    mean::Union{Float64, Missing}
    var::Union{Float64, Missing}
    std::Union{Float64, Missing}
end

function simulate!(markovchain, periods=1000, burn=0, replications=0; random_state=-1)
    if random_state >= 0
        random.seed!(random_state)
    end
    N = length(markovchain.states)
    
    # Configure CDF
    Θ = Array{dist.Categorical{Float64, Vector{Float64}}}(undef, N)
    for i = 1:N
        Θ[i] = dist.Categorical(markovchain.theta[i, :])
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
        state_series[t] = markovchain.states[index_series[t]]
    end

    markovchain.mean = stats.mean(state_series)
    markovchain.var = stats.var(state_series)
    markovchain.std = stats.std(state_series)

    # If replication is specified, rerun simulation and take average of moments
    if replications > 0
        for replication = 1:replications
            index_series[1] = rand(1:N)
            if burn > 0
                for _ = 1:burn
                    index_series[1] = dist.rand(Θ[index_series[1]], 1)[1]
                end
            end

            for t = 2:periods
                index_series[t] = dist.rand(Θ[index_series[t-1]], 1)[1]
            end
            
            for t = 1:periods
                state_series[t] = markovchain.states[index_series[t]]
            end

            markovchain.mean += stats.mean(state_series)
            markovchain.var += stats.var(state_series)
            markovchain.std += stats.std(state_series)
        end
        markovchain.mean /= (replications + 1)
        markovchain.var /= (replications + 1)
        markovchain.std /= (replications + 1)
    end

    return state_series
end

#==========================================================================================
# Tauchen / Rouwenhorst
==========================================================================================#
function tauchen(
    mean,
    variance,
    N,
    m,
    rho;
    print_output=false
    )
    
    std = sqrt(variance)
    states = Array{Float64}(undef, N)
    states[N] = m*sqrt(variance / (1 - rho^2))
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
    Θ = Array{Float64}(undef, (N, N))
    for i = 1:N
        for j = 1:N
            if j == 1
                Θ[i, j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/std)
            elseif j == N
                Θ[i, j] = 1 - dist.cdf(D, (states[j] - d/2 - rho*states[i])/std)
            else
                Θ[i, j] = dist.cdf(D, (states[j] + d/2 - rho*states[i])/std) -
                    dist.cdf(D, (states[j] - d/2 - rho*states[i])/std)
            end
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[i, :], digits=3))
        end
    end
    
    states .+= mean / (1 - rho)

    return MarkovChain(Θ, states, missing, missing, missing)
end

function rouwenhorst(
    mean,
    variance,
    N,
    rho;
    print_output=false
    )

    ψ = sqrt(variance / (1 - rho^2)) * sqrt(N - 1)

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
    Θₙ = Array{Float64}(undef, (2, 2))
    Θₙ[1, 1] = p
    Θₙ[1, 2] = 1 - p
    Θₙ[2, 1] = 1 - q
    Θₙ[2, 2] = q
    Θ = Θₙ
    if N > 2
        global Θ
        global Θₙ
        for n = 3:N
            global Θ
            global Θₙ
            Θ = zeros((n, n))
            Θ[1:n-1, 1:n-1] += p .* Θₙ
            Θ[1:n-1, 2:n] += (1 - p) .* Θₙ
            Θ[2:n, 1:n-1] += (1 - q) .* Θₙ
            Θ[2:n, 2:n] += q .* Θₙ
            Θ[2:n-1, :] ./= 2
            Θₙ = Θ
        end
    end
    
    if print_output
        println("Θ:")
        for i = 1:N
            println(round.(Θ[i, :], digits=3))
        end
    end

    states .+= mean / (1 - rho)
    
    return MarkovChain(Θ, states, missing, missing, missing)
end

#==============================================================================
# Problem 1 Results
==============================================================================#
println("\n\nProblem 1a)")
markov = tauchen(w̄, vₑ, 5, 3, ρ; print_output=false)
yts5 = simulate!(markov, 10000, 1000, 100; random_state=42)
println("Sample moments (T5):")
println("Mean: ", round(markov.mean, digits=3))
println("Variance: ", round(markov.var, digits=3))
println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\n\nProblem 1b)")
markov = tauchen(w̄, vₑ, 11, 3, ρ; print_output=false)
yts11 = simulate!(markov, 10000, 1000, 100; random_state=42)
println("Sample moments (T11):")
println("Mean: ", round(markov.mean, digits=3))
println("Variance: ", round(markov.var, digits=3))
println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\n\nProblem 1c)")
markov = rouwenhorst(w̄, vₑ, 5, ρ; print_output=false)
yrs5 = simulate!(markov, 10000, 1000, 100; random_state=42)
println("Sample moments (R5):")
println("Mean: ", round(markov.mean, digits=3))
println("Variance: ", round(markov.var, digits=3))
markov = tauchen(w̄, vₑ, 5, 3, ρ; print_output=false)
yts5 = simulate!(markov, 10000, 1000, 100; random_state=42)
println("\nSample moments (T5):")
println("Mean: ", round(markov.mean, digits=3))
println("Variance: ", round(markov.var, digits=3))
println("\nTheoretical moments:")
println("Mean: ", round(tmean, digits=3))
println("Variance: ", round(tvar, digits=3))

println("\n\nProblem 1d)")
markov = tauchen(w̄98, vₑ98, 5, 3, ρ98; print_output=false)
yts5p = simulate!(markov, 10000, 1000, 100; random_state=42)
println("Sample moments (T5, ρ=0.98):")
println("Mean: ", round(markov.mean, digits=3))
println("Variance: ", round(markov.var, digits=3))
markov = rouwenhorst(w̄98, vₑ98, 5, ρ98; print_output=false)
yrs5p = simulate!(markov, 10000, 1000, 100; random_state=42)
println("\nSample moments (R5, ρ=0.98):")
println("Mean: ", round(markov.mean, digits=3))
println("Variance: ", round(markov.var, digits=3))
println("\nTheoretical moments (ρ=0.98):")
println("Mean: ", round(tmean98, digits=3))
println("Variance: ", round(tvar98, digits=3))

#==============================================================================
# Sanity Check
==============================================================================#
mc5qe = qe.tauchen(5, ρ, σₑ, w̄, 3)
yts5qe = qe.simulate(mc5qe, 10000)
mc11qe = qe.tauchen(11, ρ, σₑ, w̄, 3)
yts11qe = qe.simulate(mc11qe, 10000)

mcr5qe = qe.rouwenhorst(5, ρ, σₑ, w̄)
yrs5qe = qe.simulate(mcr5qe, 10000)
mcr598qe = qe.rouwenhorst(5, 0.98, σₑ98, w̄98)
yrs598qe = qe.simulate(mcr598qe, 10000)

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
