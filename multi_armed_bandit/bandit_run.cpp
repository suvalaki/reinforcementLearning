#include <iostream>
#include <iomanip>
#include "bandit.hpp"

int main(int argc, char *argv[])
{
    char *tmp;
    std::size_t n_bandits = static_cast<std::size_t>(strtol(argv[1], &tmp, 10));
    std::size_t n_samples = static_cast<std::size_t>(strtol(argv[2], &tmp, 10));

    // setup bandits
    std::minstd_rand generator;
    NormalPriorNormalMultiArmedBandit<double> bandits(n_bandits, generator);

    for (size_t i = 0; i < n_samples; i++)
    {
        if (i == 0)
        {
            std::cout << "Generatimng Ranom Numbers"
                      << " with " << n_bandits << " bandits and " << n_samples << " samples\n";
        }
        else
        {
            std::cout << "\n";
        }
        std::vector<double> result = bandits.sample(generator);
        for (auto const &v : result)
        {
            std::cout << std::setw(9) << v << " ";
        }
    }
    std::cout << "\n";
    return 0;
}