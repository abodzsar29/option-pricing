#include <algorithm>    // For std::max
#include <cmath>        // For exp and sqrt
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>       // runtime measurement

static double drift(const double& r, const double& v, const double& T) {
    return T * (r - 0.5 * v * v);
}

static double diffusion(const double& v, const double& T) {
    return v * sqrt(T);
}

static double monteCarloPricing(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T, bool isCall) {
    double payoffSum = 0.0;

    double localDrift = drift(r, v, T);
    double localDiffusion = diffusion(v, T);

#pragma omp parallel reduction(+:payoffSum)  // allows multi-threading and correct accumulation of the payoffSum variable across multiple threads
    {
        std::random_device rd;  // Random number generation is inside the same function as the simulations
        std::mt19937 generator(rd());  // tp prevent a further function call per each iteration
        std::normal_distribution<> distribution(0.0, 1.0);

#pragma omp for schedule(static)  // Static mode allocates (num_sims / number of threads) number of simulations to each thread
        for (int i = 0; i < num_sims; ++i) {
            double gauss_bm = distribution(generator);
            double currentPayoff = S * exp(localDrift + (localDiffusion * gauss_bm));

            if (isCall) {
                payoffSum += std::max(currentPayoff - K, 0.0);
            }
            else {
                payoffSum += std::max(K - currentPayoff, 0.0);
            }
        }
    }

    return (payoffSum / static_cast<double>(num_sims)) * exp(-r * T);
}

int main() {
    const int num_sims = 1000000;  // number of simulations
    const double K = 100.0;        // strike price
    const double S = 100.0;        // option price
    const double v = 0.2;          // volatility: 20%
    const double r = 0.05;         // risk free rate: 5%
    const double T = 1.0;          // expiry: 1 year

    auto start = std::chrono::high_resolution_clock::now();

    double call = monteCarloPricing(num_sims, S, K, r, v, T, true);
    double put = monteCarloPricing(num_sims, S, K, r, v, T, false);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Price of Call: " << call << std::endl;
    std::cout << "Price of Put: " << put << std::endl;
    std::cout << "Runtime: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}