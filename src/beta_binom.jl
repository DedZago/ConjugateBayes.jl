# Conjugates for
#
#   Beta - Bernoulli
#   Beta - Binomial
#

update_parameters(pri::Beta, ss::BernoulliStats) = (pri.α + ss.cnt1, pri.β + ss.cnt0)
update_parameters(pri::Beta, ss::BinomialStats) = (pri.α + ss.ns, pri.β + (ss.ne * ss.n - ss.ns))
update_parameters(pri::Gamma, ss::PoissonStats) = (pri.α + ss.sx, pri.θ/(ss.tw*pri.θ + 1.0))

posterior_canon(pri::Beta, ss::BernoulliStats) = Beta(update_parameters(pri, ss)...)
posterior_canon(pri::Beta, ss::BinomialStats) = Beta(update_parameters(pri, ss)...)
posterior_canon(pri::Gamma, ss::PoissonStats) = Gamma(update_parameters(pri, ss)...)

function predictive_canon(pri::Beta, ss::BernoulliStats)
	pars = update_parameters(pri, ss)
	return Bernoulli(pars[1]/(pars[1] + pars[2]))
end

function predictive_canon(pri::Beta, ss::BinomialStats)
	pars = update_parameters(pri, ss)
	return BetaBinomial(ss.n, pars[1], pars[2])
end

function predictive_canon(pri::Gamma, ss::PoissonStats)
	pars = update_parameters(pri, ss)
	return NegativeBinomial(pars[1], pars[2]/(pars[2] + 1.0))
end


complete(G::Type{Bernoulli}, pri::Beta, p::Float64) = Bernoulli(p)

# specialized fit_map and posterior_randmodel methods for Binomial
#
# n is needed to create a Binomial distribution (which can not be provided through complete)
#

fit_map(pri::Beta, ss::BinomialStats) = Binomial(ss.n, posterior_mode(pri, ss))
fit_map(pri::Beta, G::Type{Binomial}, data::BinomData) = fit_map(pri, suffstats(G, data))
fit_map(pri::Beta, G::Type{Binomial}, data::BinomData, w::Array) = fit_map(pri, suffstats(G, data, w))

posterior_randmodel(pri::Beta, ss::BinomialStats) = Binomial(ss.n, posterior_rand(pri, ss))

function posterior_randmodel(pri::Beta, G::Type{Binomial}, data::BinomData)
	posterior_randmodel(pri, suffstats(G, data))
end

function posterior_randmodel(pri::Beta, G::Type{Binomial}, data::BinomData, w::Array)
	posterior_randmodel(pri, suffstats(G, data, w))
end
