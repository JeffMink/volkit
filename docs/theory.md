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

### Estimator

The price is a risk-neutral expectation, $V = e^{-rT}\mathbb{E}^{\mathbb{Q}}[g(S_T)]$,
so it can be estimated by a sample mean. Because $\ln S_T$ is normal, $S_T$ is
sampled exactly in one step,

$$S_T = S\exp\!\left((r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}\,Z\right),
\qquad Z \sim \mathcal{N}(0,1),$$

with no time-discretisation error (path simulation is only needed for
path-dependent payoffs). With $X_i = e^{-rT} g(S_T^{(i)})$,

$$\hat V = \frac1n \sum_{i=1}^n X_i, \qquad
\text{SE}(\hat V) = \frac{s_X}{\sqrt n}.$$

By the central limit theorem the error is about $\text{SE}$, and it falls like
$1/\sqrt n$: four times the paths for half the error. Variance reduction lowers
the constant $s_X$ instead.

### Antithetic variates

If $Z \sim \mathcal{N}(0,1)$ then so is $-Z$. Use each draw twice and average:
$\bar X_i = \tfrac12\left(X(Z_i) + X(-Z_i)\right)$. Then

$$\operatorname{Var}(\bar X_i) = \tfrac12\operatorname{Var}(X)\,(1 + \rho),
\qquad \rho = \operatorname{Corr}(X(Z), X(-Z)).$$

Option payoffs are monotone in $Z$, so $\rho < 0$ and the pair average beats
two independent draws. The pairs are independent, so the standard error is
computed from the $n/2$ pair averages, not the $n$ individual payoffs.

### Control variate

Take a quantity $Y$, correlated with $X$, whose mean is known exactly. Here
$Y = e^{-rT}S_T$, with $\mathbb{E}^{\mathbb{Q}}[Y] = S e^{-qT}$ (the discounted
stock is a martingale). For any $\beta$, $X - \beta(Y - \mathbb{E}Y)$ is still
unbiased, and its variance $\operatorname{Var}X - 2\beta\operatorname{Cov}(X,Y)
+ \beta^2\operatorname{Var}Y$ is minimised at

$$\beta^* = \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}, \qquad
\operatorname{Var}_{\min} = (1 - \rho_{XY}^2)\operatorname{Var}(X).$$

$\beta^*$ is the slope of a regression of $X$ on $Y$, estimated from the same
sample; this adds a bias of order $1/n$, negligible next to the standard error.
The control works best when the payoff is close to linear in $S_T$ (deep ITM
calls). A consequence worth knowing: since $P = C - (Y - Ke^{-rT})$ pathwise,
the put and the call are the same estimator after the control-variate
adjustment and have identical standard errors.

## Binomial tree

### CRR construction

Split $[0, T]$ into $n$ steps of length $\Delta t = T/n$. Each step the price
goes up by $u$ or down by $d$, with risk-neutral probability $p$ of an up-move.
Cox, Ross and Rubinstein (1979) choose

$$u = e^{\sigma\sqrt{\Delta t}}, \qquad d = 1/u, \qquad
p = \frac{e^{(r-q)\Delta t} - d}{u - d}.$$

- **Mean:** $p$ is chosen so that $pu + (1-p)d = e^{(r-q)\Delta t}$, i.e. the
  discounted stock (with dividends reinvested) is a martingale on the tree.
  This is what makes the tree arbitrage-free, and it makes put-call parity hold
  exactly on the tree for every $n$.
- **Variance:** the one-step variance of the log-return is
  $\sigma^2\Delta t + O(\Delta t^2)$, matching Black-Scholes.
- $d = 1/u$ makes the tree recombine: up then down returns to the same price,
  so step $i$ has only $i+1$ nodes, $S_{i,j} = S u^j d^{i-j}$.
- $p \in (0,1)$ requires $d < e^{(r-q)\Delta t} < u$; it fails only if
  $\Delta t$ is too large, and volkit then raises an error.

### Backward induction

At expiry $V_{n,j} = g(S_{n,j})$. Each earlier node is the discounted
risk-neutral expectation of its two successors:

$$V_{i,j} = e^{-r\Delta t}\left[p\,V_{i+1,j+1} + (1-p)\,V_{i+1,j}\right].$$

The cost is $O(n^2)$; storing only one time layer keeps memory at $O(n)$.

### Convergence

By the central limit theorem the tree's terminal log-price converges to the
normal distribution of Black-Scholes, and the European price converges with
error $O(1/n)$. The error oscillates in $n$ because the strike sits at a
different position relative to the terminal nodes at each $n$; for at-the-money
options with even $n$ it is smooth and halves when $n$ doubles. The tests
check both.

### American exercise

An American option can be exercised at any time, so its holder compares the
value of exercising now with the value of waiting, at every node:

$$V_{i,j} = \max\!\left(g(S_{i,j}),\;
e^{-r\Delta t}\left[p\,V_{i+1,j+1} + (1-p)\,V_{i+1,j}\right]\right).$$

This is dynamic programming for an optimal stopping problem. The nodes where
the first term wins form the early-exercise region; its boundary is the
critical stock price $S^*(t)$.

- **Calls, no dividends:** $C \ge S - Ke^{-rT} \ge S - K$ when $r \ge 0$, so
  the option is always worth more alive than exercised. The American call
  equals the European call (Merton, 1973).
- **Calls with dividends:** holding the stock earns $q$; exercising a deep ITM
  call early to capture it can be optimal.
- **Puts:** exercising early gets $K$ now and earns interest on it, so a deep
  ITM put is exercised early whenever $r > 0$. The American put has no closed
  form, which is why numerical methods matter.

## Market data and the raw surface

The surface is built from one snapshot of SPX option quotes taken at time
$t_0$. SPX options are European, so Black-Scholes implied vols carry no
early-exercise premium.

### Time to expiry

$T = (\text{expiry} - t_0) / 365$ days (ACT/365), with the expiry including
its settlement time: standard SPX options settle at the open (09:30 ET),
SPXW weeklies at the close (16:00 ET). For a one-week option the 6.5-hour gap
is about 4% of $T$, so it matters. $t_0$ is the time of the index's last
price, so quotes taken after the close are not given extra time.

### Forward and discount factor from put-call parity

For each expiry, put-call parity gives a model-free line in the strike:

$$C(K) - P(K) = D\,(F - K), \qquad D = e^{-rT}.$$

A least-squares fit $C - P = a + bK$ gives $D = -b$ and $F = -a/b$. This
uses the market's own forward, so the dividend yield never has to be
guessed. Errors in $q$ would otherwise show up directly as skew in the
surface.

**Which strikes.** Only the 20 strikes closest to the spot price enter the
fit. Far from the money, quotes can be stale: the real snapshot had a call
with strike 400 quoted at 0.02 when it was worth about 6,300. Choosing
strikes by smallest $|C-P|$ picked that pair and broke the fit. The spot only
has to locate the money roughly.

**Short expiries.** The implied rate $r = -\ln D / T$ turns an error
$\delta D$ into $\delta r \approx \delta D / T$, so for $T = 0.02$ a 0.2%
error in $D$ is a 10% error in $r$. For $T < 0.1$ volkit instead uses the
median rate $\bar r$ of the longer expiries, sets $D = e^{-\bar r T}$, and
refits only the forward. Each strike then gives $F = K + (C-P)/D$, and
these are averaged:

$$F = \frac{1}{n}\sum_i \left[K_i + \frac{C_i - P_i}{D}\right].$$

Expiries whose free fit gives a rate outside $[-2\%, 15\%]$ are dropped as
broken.

### Cleaning

Quotes are kept only if $\text{bid} > 0$, $\text{ask} \ge \text{bid}$, the
spread is at most 50% of the mid, and the option is out of the money (puts
with $K < F$, calls with $K \ge F$). By parity an ITM option carries the
same information as the OTM option at its strike, but with a wider spread.
Expiries under 7 days, and expiries left with fewer than 10 quotes, are
dropped.

### Implied vol with a known forward

With $F$ and $D$ known, the Black (1976) price is

$$C = D\left[F\,N(d_1) - K\,N(d_2)\right], \qquad
d_{1,2} = \frac{\ln(F/K) \pm \tfrac12\sigma^2 T}{\sigma\sqrt T}.$$

This is Black-Scholes-Merton with $S = F$ and $r = q = -\ln D / T$, because
then $Se^{-qT} = FD$ and $Ke^{-rT} = KD$. So the existing solver inverts it
unchanged.

### Coordinates for SVI

Each quote is stored as log-moneyness $k = \ln(K/F)$ and total implied
variance $w = \sigma^2 T$, the variables SVI is written in.

## SVI parametrisation

## Static arbitrage conditions

## References

- F. Black and M. Scholes (1973). The Pricing of Options and Corporate
  Liabilities. *Journal of Political Economy* 81(3).
- R. C. Merton (1973). Theory of Rational Option Pricing. *Bell Journal of
  Economics and Management Science* 4(1).
- J. C. Cox, S. A. Ross and M. Rubinstein (1979). Option Pricing: A
  Simplified Approach. *Journal of Financial Economics* 7(3).
- P. Glasserman (2003). *Monte Carlo Methods in Financial Engineering*.
  Springer.
- F. Black (1976). The Pricing of Commodity Contracts. *Journal of Financial
  Economics* 3(1-2).
- S. Manaster and G. Koehler (1982). The Calculation of Implied Variances from
  the Black-Scholes Model: A Note. *Journal of Finance* 37(1).
- P. Jäckel (2015). Let's Be Rational. *Wilmott*, January 2015.
- J. C. Hull. *Options, Futures, and Other Derivatives*. Pearson.
