#include <cmath>

static constexpr int dim = 8;
static constexpr double lb = -10.0;
static constexpr double ub = 10.0;
// Optimum: f(x) = 0, x = (1, 1, ...)

double f(const std::vector<double> & x)
{
    double res = 0.0;

    std::vector<double> w(dim);
    for (std::size_t i = 0; i < dim; ++i)
        w[i] = 1.0 + (x[i] - 1.0) / 4.0;

    double term1 = std::pow(std::sin(M_PI * w[0]), 2.0);
    double term3 = std::pow(w[dim - 1] - 1.0, 2.0) *
                   (1.0 + std::pow(std::sin(2.0 * M_PI * w[dim - 1]), 2.0));

    double term2 = 0.0;
    for (std::size_t i = 0; i < dim - 1; ++i)
    {
        double wi = w[i];
        term2 += std::pow(wi - 1.0, 2.0) *
                 (1.0 + 10.0 * std::pow(std::sin(M_PI * wi + 1.0), 2.0));
    }

    res = term1 + term2 + term3;
    return res;
}
