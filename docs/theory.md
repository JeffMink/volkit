# Theory

Derivations behind the methods in volkit. Each section is filled in when the
corresponding feature is implemented.

Notation: spot $S$, strike $K$, time to expiry $T$ (years), risk-free rate $r$,
dividend yield $q$, volatility $\sigma$. $N$ is the standard normal CDF and
$\varphi$ its density.

## Black-Scholes and Greeks

### Model

Under the risk-neutral measure $\mathbb{Q}$ the stock follows geometric
Brownian motion with drift $r - q$:

$$dS_t = (r - q)\,S_t\,dt + \sigma S_t\,dW_t.$$

By Itô's lemma $\ln S_T$ is normal:

$$\ln S_T \sim \mathcal{N}\!\left(\ln S + (r - q - \tfrac12\sigma^2)T,\; \sigma^2 T\right).$$

### Price

A European option is worth its discounted expected payoff under $\mathbb{Q}$.
For a call, $C = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[(S_T - K)^+]$. Splitting the
expectation into $\mathbb{E}[S_T \mathbf{1}_{S_T > K}]$ and
$K\,\mathbb{Q}(S_T > K)$ and evaluating both lognormal integrals gives

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),$$

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \qquad
d_2 = d_1 - \sigma\sqrt{T}.$$

Interpretation: $N(d_2) = \mathbb{Q}(S_T > K)$ is the risk-neutral probability
of exercise, and $S e^{-qT} N(d_1)$ is the present value of receiving the stock
in exactly those outcomes.

The same formula follows from the hedging argument: a portfolio of the option
and $-\Delta$ shares is riskless, so it must earn $r$, which gives the
Black-Scholes PDE

$$\Theta + (r - q) S \Delta + \tfrac12 \sigma^2 S^2 \Gamma - rV = 0.$$

The test suite checks that the closed-form price and Greeks satisfy this PDE.

### Put-call parity

A long call and a short put with the same strike pay $S_T - K$ at expiry,
whatever happens. So, with no model at all,

$$C - P = S e^{-qT} - K e^{-rT}.$$

### Greeks

Differentiating the price formula, and using the identity
$S e^{-qT} \varphi(d_1) = K e^{-rT} \varphi(d_2)$ to cancel terms:

| Greek | Definition | Call | Put |
|---|---|---|---|
| Delta | $\partial V/\partial S$ | $e^{-qT}N(d_1)$ | $-e^{-qT}N(-d_1)$ |
| Gamma | $\partial^2 V/\partial S^2$ | $\dfrac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt{T}}$ | same |
| Vega | $\partial V/\partial \sigma$ | $S e^{-qT}\varphi(d_1)\sqrt{T}$ | same |
| Rho | $\partial V/\partial r$ | $K T e^{-rT} N(d_2)$ | $-K T e^{-rT} N(-d_2)$ |

Theta, $\Theta = \partial V/\partial t = -\partial V/\partial T$:

$$\Theta_C = -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}} - rK e^{-rT}N(d_2) + qS e^{-qT}N(d_1)$$

$$\Theta_P = -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt{T}} + rK e^{-rT}N(-d_2) - qS e^{-qT}N(-d_1)$$

Gamma and vega are the same for calls and puts because, by put-call parity,
$C - P$ is linear in $S$ and does not depend on $\sigma$.

volkit returns raw derivatives: vega per 1.00 of volatility (divide by 100 for
"per vol point") and theta per year (divide by 365 for "per calendar day").

## Implied volatility

### Definition and uniqueness

The implied volatility of an observed price $V^{\text{mkt}}$ is the $\sigma$
that solves $\text{BS}(\sigma) = V^{\text{mkt}}$.

Since vega $> 0$, $\text{BS}(\sigma)$ is strictly increasing. As
$\sigma \to 0$ it tends to the discounted intrinsic value, and as
$\sigma \to \infty$ to $S e^{-qT}$ (call) or $K e^{-rT}$ (put). So a solution
exists, and is unique, exactly when the price lies strictly inside the
no-arbitrage bounds:

$$\max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}, \qquad
\max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}.$$

Market quotes outside these bounds (stale quotes, wide spreads) have no implied
volatility; volkit returns NaN for them.

### Algorithm: safeguarded Newton-Raphson

Newton's method on $f(\sigma) = \text{BS}(\sigma) - V^{\text{mkt}}$:

$$\sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - V^{\text{mkt}}}{\text{vega}(\sigma_n)}.$$

It converges quadratically near the root, but for deep out-of-the-money options
vega is tiny and a raw Newton step can shoot far away. volkit therefore keeps a
bracket $[\sigma_{lo}, \sigma_{hi}]$ with $f(\sigma_{lo}) < 0 < f(\sigma_{hi})$,
updated from the sign of $f$ at each iterate. If a Newton step would leave the
bracket, or vega is too small, it bisects instead. The bracket shrinks every
iteration, so the method cannot diverge.

- **Start:** $\sigma_0 = \sqrt{2|\ln(F/K)|/T}$ with forward
  $F = S e^{(r-q)T}$. This is the inflection point of $\text{BS}(\sigma)$
  (Manaster and Koehler, 1982), from which Newton converges monotonically.
  It is floored at 10% so at-the-money options do not start at zero.
- **Stopping rule:** stop when $\sigma$ is known to within $10^{-12}$, from
  the Newton step size or the bracket width. A tolerance on $\sigma$ rather
  than on price matters: the error in $\sigma$ is roughly the price error
  divided by vega, so for options with tiny vega a tiny price error is a large
  volatility error.

### Accuracy limit

Floating-point prices carry a relative rounding error of about $10^{-16}$, so
no solver can recover $\sigma$ better than about
$10^{-16}\,V / \text{vega}$. The tests check that $\sigma$ is recovered to
$10^{-7}$ wherever vega $> 10^{-6}$, and that every returned volatility
reprices the option to within $10^{-12}$.

## Monte Carlo pricing

## Binomial tree

## SVI parametrisation

## Static arbitrage conditions

## References

- F. Black and M. Scholes (1973). The Pricing of Options and Corporate
  Liabilities. *Journal of Political Economy* 81(3).
- R. C. Merton (1973). Theory of Rational Option Pricing. *Bell Journal of
  Economics and Management Science* 4(1).
- S. Manaster and G. Koehler (1982). The Calculation of Implied Variances from
  the Black-Scholes Model: A Note. *Journal of Finance* 37(1).
- P. Jäckel (2015). Let's Be Rational. *Wilmott*, January 2015.
- J. C. Hull. *Options, Futures, and Other Derivatives*. Pearson.
