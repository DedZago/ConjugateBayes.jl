# Conjuagtes for
#
#	Gamma - Exponential
#


update_parameters(prior::Gamma, ss::ExponentialStats) = (prior.α + ss.sw, 1.0 / (prior.θ + ss.sx))

function posterior_canon(prior::Gamma, ss::ExponentialStats)
	return Gamma(update_parameters(prior, ss)...)
end

function predictive(prior::Gamma, ss::ExponentialStats)
	#! TODO: Lomax distribution
	pars = update_parameters(prior, ss)
	return Pareto(pars...)
end

complete(G::Type{Exponential}, pri::Gamma, θ::Float64) = Exponential(1.0 / θ)
