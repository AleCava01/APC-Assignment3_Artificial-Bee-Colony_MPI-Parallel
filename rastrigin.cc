#include <cmath>

static constexpr int dim = 16;
static constexpr double lb = -5.12;
static constexpr double ub = 5.12;
// Optimum: f(x) = 0, x = (0, 0, ...)

double f(const std::vector<double> & x){

    double res = 10 * dim;

    for (std::size_t ii = 0; ii < dim; ++ii)
        res += std::pow(x[ii],2) - 10*cos(2 * M_PI * x[ii]);

    return res;
}