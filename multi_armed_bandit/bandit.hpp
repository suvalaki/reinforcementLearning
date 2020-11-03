#include <random>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>

template <class TYPE_T>
class Bandit
{
public:
    virtual TYPE_T sample(std::minstd_rand &engine) { return static_cast<TYPE_T>(NULL); };
    virtual TYPE_T value() { return qA; }

private:
    TYPE_T qA;
};

template <typename TYPE_T>
class MultiArmedBandit
{
public:
    Bandit<TYPE_T> &getBandit(size_t i) { return &bandits[i]; };
    std::size_t getBestBanditIdx()
    {
        std::size_t bestIdx = 0;
        TYPE_T bestValue = std::numeric_limits<TYPE_T>::min();
        for (std::size_t i = 0; i < bandits.size(); i++)
        {
            if (bandits[i].value() > bestValue)
                bestIdx = i;
        }
        return bestIdx;
    }
    Bandit<TYPE_T> &getBestBandit() { return &getBestBanditIdx(); }
    TYPE_T getBestValue() { return getBestBandit().value(); }
    virtual std::vector<TYPE_T> sample(std::minstd_rand &engine) { return std::vector<TYPE_T>(); };

private:
    std::vector<Bandit<TYPE_T>> bandits;
    std::size_t n_bandits;
};

template <class TYPE_T>
class NormalBandit : public Bandit<TYPE_T>
{

public:
    NormalBandit(TYPE_T avg = 0, TYPE_T stddev = 1) : qA(avg), avg(avg), stddev(stddev), distribution(avg, stddev){};
    TYPE_T sample(std::minstd_rand &engine)
    {
        TYPE_T result;
        do
        {
            result = distribution(engine);
        } while (result <= min);
        return result;
    };
    friend std::ostream &operator<<(std::ostream &os, const NormalBandit<TYPE_T> &bandit)
    {
        os << "NormalBandit(avg = " << bandit.avg << ", stddev = " << bandit.stddev << ")";
        return os;
    }

private:
    std::normal_distribution<double> distribution;
    TYPE_T qA = 0;
    TYPE_T avg = 0;
    TYPE_T stddev = 0;
    TYPE_T min = 0;
};

template <typename TYPE_T>
class NormalPriorNormalMultiArmedBandit : public MultiArmedBandit<TYPE_T>
{
public:
    NormalPriorNormalMultiArmedBandit(
        std::size_t n_bandits,
        std::minstd_rand &engine,
        TYPE_T prior_mu_ave = 0, TYPE_T prior_mu_stddev = 1,
        TYPE_T prior_var_ave = 0, TYPE_T prior_var_stddev = 1) : n_bandits(n_bandits)
    {
        this->bandits.reserve(n_bandits);

        std::normal_distribution<TYPE_T> prior_mu(prior_mu_ave, prior_mu_stddev);
        std::normal_distribution<TYPE_T> prior_var(prior_var_ave, prior_var_stddev);

        // calculate means
        std::vector<TYPE_T> means(n_bandits);
        std::vector<TYPE_T> stddevs(n_bandits);
        for (std::size_t i = 0; i < n_bandits; i++)
        {
            do
            {
                means[i] = prior_mu(engine);
            } while (means[i] <= 0);
            do
            {
                stddevs[i] = prior_var(engine);
            } while (stddevs[i] <= 0);
            this->bandits.emplace_back(NormalBandit<TYPE_T>(means[i], stddevs[i]));
        }
    };

    std::vector<TYPE_T> sample(std::minstd_rand &engine)
    {
        std::vector<TYPE_T> samples(n_bandits);
        for (int i = 0; i < n_bandits; i++)
        {
            samples[i] = this->bandits[i].sample(engine);
        }
        return samples;
    };

private:
    std::vector<NormalBandit<TYPE_T>> bandits;
    std::size_t n_bandits;
};
