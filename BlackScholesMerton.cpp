#include <iostream>
#include <cmath>
#include <chrono>  // runtime measurement

static double d1(const double& S, const double& K, const double& r, const double& v, const double& T) {
    return (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
}

static double d2(const double& d1, const double& v, const double& T) {
    return d1 - v * std::sqrt(T);
}

// Cumulative Distribution Function (CDF) of a normal distribution
static double normalCDF(const double& x) {
    return 0.5 * std::erfc(-x / std::sqrt(2));
}

static double callPrice(const double& S, const double& K, const double& r, const double& v, const double& T, const double& d1, const double& d2) {
    return S * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
}

static double putPrice(const double& S, const double& K, const double& r, const double& v, const double& T, const double& d1, const double& d2) {
    return K * exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
}


int main() {
    const double K = 100.0;  // strike price
    const double S = 100.0;  // spot price
    const double v = 0.2;    // volatility: 20%
    const double r = 0.05;   // risk-free rate: 5%
    const double T = 1.0;    // expiry: 1 year

    auto start = std::chrono::high_resolution_clock::now();

    double d1Value = d1(S, K, r, v, T);
    double d2Value = d2(d1Value, v, T);

    double call = callPrice(S, K, r, v, T, d1Value, d2Value);
    double put = putPrice(S, K, r, v, T, d1Value, d2Value);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Price of Call: " << call << std::endl;
    std::cout << "Price of Put: " << put << std::endl;
    std::cout << "Runtime: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}