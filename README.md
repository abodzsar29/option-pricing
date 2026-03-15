# Option Pricing – Black-Scholes-Merton, Binomial Tree & Monte Carlo

Three independent **C++** implementations of classical option pricing models for European-style call and put options.

---

## Overview

This project provides side-by-side implementations of the three most widely used derivative pricing methodologies, making it straightforward to compare their approaches, assumptions, and computational characteristics.

| Source file | Model | Approach |
|---|---|---|
| `BlackScholesMerton.cpp` | Black-Scholes-Merton | Closed-form analytical solution |
| `Binomial.cpp` | Cox-Ross-Rubinstein Binomial Tree | Discrete-time backward induction |
| `MonteCarlo.cpp` | Monte Carlo Simulation | Stochastic path simulation with OpenMP parallelism |

All three models price vanilla **call** and **put** options and report wall-clock **runtime** for performance comparison.

---

## Models

### Black-Scholes-Merton (`BlackScholesMerton.cpp`)

Computes exact analytical prices for European call and put options using the BSM closed-form formulae. The implementation calculates the intermediate terms *d₁* and *d₂*, evaluates the **cumulative normal distribution** via `std::erfc`, and applies the standard discounted expectation formulae for each option type.

**Parameters used in the example:**

| Parameter | Value | Description |
|---|---|---|
| `S` | 100.0 | Spot price |
| `K` | 100.0 | Strike price |
| `v` | 0.20 | Annualised volatility (20%) |
| `r` | 0.05 | Continuously compounded risk-free rate (5%) |
| `T` | 1.0 | Time to expiry (1 year) |

---

### Binomial Tree (`Binomial.cpp`)

Prices options using the **Cox-Ross-Rubinstein (CRR)** binomial lattice via backward induction. The tree is constructed to be consistent with **Geometric Brownian Motion (GBM)**: the up-factor `u = exp(σ√Δt)` and down-factor `d = 1/u` produce a symmetrical tree, and the risk-neutral probability `p` is calibrated so that the expected asset growth matches the risk-free rate, enforcing the no-arbitrage condition. Terminal payoffs are computed at the final nodes and discounted back step-by-step to the root.

The `BinomialOptionPricing` class encapsulates the full pricing workflow and exposes a single public method `calculateOptionPrice(bool isCall)`.

**Parameters used in the example:**

| Parameter | Value | Description |
|---|---|---|
| `S` | 100.0 | Spot price |
| `K` | 101.0 | Strike price |
| `sigma` | 0.20 | Annualised volatility (20%) |
| `r` | 0.05 | Risk-free rate (5%) |
| `T` | 0.5 | Time to expiry (6 months) |
| `steps` | 2 | Number of time steps in the tree |

---

### Monte Carlo Simulation (`MonteCarlo.cpp`)

Prices options by simulating a large number of independent asset price paths under the **risk-neutral measure**, computing the discounted average payoff across all paths. Each path evolves the underlying asset according to the GBM closed-form solution, sampling a single standard normal random variate per path via `std::mt19937` and `std::normal_distribution`.

The simulation is parallelised using **OpenMP** with a `parallel reduction` over the payoff accumulator and a `static` loop schedule, distributing paths evenly across available CPU threads.

**Parameters used in the example:**

| Parameter | Value | Description |
|---|---|---|
| `S` | 100.0 | Spot price |
| `K` | 100.0 | Strike price |
| `v` | 0.20 | Annualised volatility (20%) |
| `r` | 0.05 | Risk-free rate (5%) |
| `T` | 1.0 | Time to expiry (1 year) |
| `num_sims` | 1,000,000 | Number of simulated paths |

---

## Build Instructions

Each source file contains its own `main` function and can be compiled independently. Below are example commands using common toolchains.

### GCC / Clang

```bash
# Black-Scholes-Merton
g++ -std=c++14 -O2 -o bsm BlackScholesMerton.cpp

# Binomial Tree
g++ -std=c++14 -O2 -o binomial Binomial.cpp

# Monte Carlo (OpenMP required)
g++ -std=c++14 -O2 -fopenmp -o montecarlo MonteCarlo.cpp
```

### MSVC (Developer Command Prompt)

```bash
# Black-Scholes-Merton
cl /EHsc /O2 BlackScholesMerton.cpp

# Binomial Tree
cl /EHsc /O2 Binomial.cpp

# Monte Carlo (OpenMP required)
cl /EHsc /O2 /openmp MonteCarlo.cpp
```

> **Note:** The Monte Carlo model depends on **OpenMP**. On GCC/Clang pass `-fopenmp`; on MSVC pass `/openmp`. OpenMP is available natively in all three toolchains.

---

## Output Example

Each executable prints the computed call and put prices together with the measured wall-clock runtime:

```
Price of Call: 10.4506
Price of Put: 5.5735
Runtime: 0.000031 seconds
```

---

## Dependencies

- **C++14** or later (all three files use only the C++ standard library, with the exception of OpenMP in the Monte Carlo model).
- **OpenMP** – required for `MonteCarlo.cpp`; available in GCC, Clang, and MSVC.
