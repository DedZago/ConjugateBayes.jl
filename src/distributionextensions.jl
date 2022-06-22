struct Lomax <: ContinuousUnivariateDistribution
    α::Float64
    λ::Float64

    function Lomax(α, λ)
        if α <= 0
            throw(DomainError(α, "Parameter should be positive"))
        elseif λ <= 0
            throw(DomainError(λ, "Parameter should be positive"))
        else
            new(Float64(α), Float64(λ)) 
        end#if
    end
end

pdf(d::Lomax, x::Float64) = Distributions.pdf(Pareto(d.α, d.λ), x + d.λ)
logpdf(d::Lomax, x::Float64) = Distributions.logpdf(Pareto(d.α, d.λ), x + d.λ)
cdf(d::UnivariateDistribution, x::Real) = Distributions.cdf(Pareto(d.α, d.λ), x + d.λ)
quantile(d::UnivariateDistribution, q::Real) = Distributions.quantile(Pareto(d.α, d.λ), q + d.λ)
minimum(d::UnivariateDistribution) = Distributions.minimum(Pareto(d.α, d.λ)) - d.λ
Distributions.maximum(d::UnivariateDistribution) = Distributions.maximum(Pareto(d.α, d.λ)) - d.λ
insupport(d::UnivariateDistribution, x::Real) = Distributions.insupport(Pareto(d.α, d.λ), x + d.λ)
Distributions.rand(d::Lomax) = Distributions.rand(Pareto(d.α, d.λ)) .- d.λ 
Distributions.rand(d::Lomax, n::Int64) = Distributions.rand(Pareto(d.α, d.λ), n) .- d.λ 