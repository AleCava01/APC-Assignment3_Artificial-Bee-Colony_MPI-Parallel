#include <cmath>

static constexpr int dim = 4;
static constexpr double lb = -500.0;
static constexpr double ub = 500.0;
// Optimum: f(x) = 0, x = (420.9687, 420.9687, ...)

double f(const std::vector<double> & x){

    double res = 418.9829 * dim;

    for (std::size_t ii = 0; ii < dim; ++ii)
        res -= x[ii]*sin(sqrt(std::abs(x[ii])));

    return res;
}