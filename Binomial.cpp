#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


// Backward Induction Method

class BinomialOptionPricing {
public:
    BinomialOptionPricing(double S, double K, double r, double T, double sigma, int steps)
        : S(S), K(K), r(r), T(T), sigma(sigma), steps(steps) {
        dt = T / steps;
        // To approximate continuous-time evolution by discrete time steps, u and d are:
        u = std::exp(sigma * std::sqrt(dt));   // To ensure consistency with GBM u takes this value
        d = 1 / u;                   // creates a symmetrical binomial tree - no move is favoured disproportionately
        p = (std::exp(r * dt) - d) / (u - d); // Ensures that expected growth of asset matches risk-free rate r - enforces no-arbitrage condition
        discount = std::exp(-r * dt);     // Discounting back for the time value of money
        optionValues.resize(steps + 1);
    }

    // Public method to drive the entire pricing process - conforming to encapsulation
    double calculateOptionPrice(bool isCall) {
        lastStepPrices(isCall);  // Step 1: Calculate terminal step payoffs
        return backStep();  // Step 2: Backward induction to compute option price at step 0
    }

private:
    // Calculate the asset prices at the last step and payoff values
    void lastStepPrices(bool isCall) {
        for (int i = 0; i <= steps; ++i) {
            double assetPrice = S * std::pow(u, steps - i) * std::pow(d, i);  // Calculating underlying price at last step
            if (isCall) {
                optionValues[i] = std::max(0.0, assetPrice - K); // Call option payoff
            }
            else {
                optionValues[i] = std::max(0.0, K - assetPrice); // Put option payoff
            }
        }
    }

    // Stepping back and calculating option prices at each step, calculating the final option price at step = 0
    double backStep() {
        for (int step = steps - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                optionValues[i] = discount * (p * optionValues[i] + (1 - p) * optionValues[i + 1]);
            }
        }
        // Return the final option value at the root step = 0
        return optionValues[0];  // Vector to store payoff values at last step; option values at previous steps
    }

    double S, K, r, T, sigma, dt, u, d, p, discount;
    int steps;
    std::vector<double> optionValues;
};

int main() {
    double S, K, r, T, sigma;
    S = 100;
    K = 101.0;
    r = 0.05;
    T = 0.5;  // Time to maturity in years
    sigma = 0.2;
    int steps = 2;

    BinomialOptionPricing callOption(S, K, r, T, sigma, steps);
    double callPrice = callOption.calculateOptionPrice(true);
    std::cout << "Price of call option: " << callPrice << std::endl;

    BinomialOptionPricing putOption(S, K, r, T, sigma, steps);
    double putPrice = putOption.calculateOptionPrice(false);
    std::cout << "Price of put option: " << putPrice << std::endl;

    return 0;
}