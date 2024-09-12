#include <algorithm>    // For std::max
#include <cmath>        // For exp and sqrt
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>       // run time measurement

double monteCarloPricing(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T, bool isCall) {
    double payoffSum = 0.0;

    const double drift = T * (r - 0.5 * v * v);
    const double diffusion = v * sqrt(T);

#pragma omp parallel reduction(+:payoffSum)
    {
        // Random Number Generator
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<> distribution(0.0, 1.0);

#pragma omp for schedule(static)
        for (int i = 0; i < num_sims; ++i) {
            double gauss_bm = distribution(generator);
            double currentPayoff = S * exp(drift + (diffusion * gauss_bm));

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
    // Parameters
    const int num_sims = 1000000;  // number of simulations
    const double K = 100.0;        // strike price
    const double S = 100.0;        // option price
    const double v = 0.2;          // volatility: 20%
    const double r = 0.05;         // risk free rate: 5%
    const double T = 1.0;          // expiry: 1 year

    // Measure the time taken to perform the simulation
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