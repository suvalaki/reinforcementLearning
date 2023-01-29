#pragma once

#include <array>
#include <iomanip>
#include <iostream>
#include <random>

#include "action.hpp"
#include "environment.hpp"
#include "iostream"
#include "policy/policy.hpp"
#include "spec.hpp"

namespace bandit {

template <std::size_t N_BANDITS>
struct BanditState : environment::State<float> {

  using BaseType = environment::State<float>;
  using typename BaseType::PrecisionType;

  std::minstd_rand &hiddenRandomEngine;
  std::array<float, N_BANDITS> hiddenBanditMeans;
  std::array<float, N_BANDITS> hiddenBanditStd;
  std::array<float, N_BANDITS> observableBanditSample;

  BanditState(std::minstd_rand &engine,
              const std::array<float, N_BANDITS> &hiddenBanditMeans,
              const std::array<float, N_BANDITS> &hiddenBanditStd,
              const std::array<float, N_BANDITS> &observableBanditSample)
      : BaseType(), hiddenRandomEngine(engine),
        hiddenBanditMeans(hiddenBanditMeans), hiddenBanditStd(hiddenBanditStd),
        observableBanditSample(observableBanditSample) {}

  BanditState static nullFactory(
      std::minstd_rand &engine,
      const std::array<float, N_BANDITS> &hiddenBanditMeans,
      const std::array<float, N_BANDITS> &hiddenBanditStd) {
    return BanditState{engine, hiddenBanditMeans, hiddenBanditStd, {0}};
  }

  BanditState operator=(const BanditState &other) {
    if (this == &other)
      return *this;
    hiddenRandomEngine = other.hiddenRandomEngine;
    hiddenBanditMeans = other.hiddenBanditMeans;
    hiddenBanditStd = other.hiddenBanditStd;
    observableBanditSample = other.observableBanditSample;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const BanditState &b) {

    os << "hiddenBanditMeans: ";
    for (const auto &r : b.hiddenBanditMeans)
      os << std::setw(10) << r << " ";

    os << "hiddenBanditStd: ";
    for (const auto &r : b.hiddenBanditStd)
      os << std::setw(10) << r << " ";

    os << "observableBanditSample: ";
    for (const auto &r : b.observableBanditSample)
      os << std::setw(10) << r << " ";

    return os;
  }

  std::size_t hash() const override { return 1; }
};

enum class BanditActionChoices { NO = 0, YES = 1 };

template <std::size_t N_BANDITS>
using BanditActionSpec = spec::CompositeArraySpec<
    spec::CategoricalArraySpec<bandit::BanditActionChoices, N_BANDITS>>;

template <std::size_t N_BANDITS>
struct BanditAction
    : action::Action<BanditState<N_BANDITS>, BanditActionSpec<N_BANDITS>> {
  using BaseType =
      action::Action<BanditState<N_BANDITS>, BanditActionSpec<N_BANDITS>>;
  using typename BaseType::BaseType;
  using typename BaseType::StateType;

  // std::array<bool, N_BANDITS> banditChoice;

  std::array<float, N_BANDITS>
  sample(std::minstd_rand &engine,
         const std::array<float, N_BANDITS> &hiddenBanditMeans,
         const std::array<float, N_BANDITS> &hiddenBanditStd) const {
    std::array<float, N_BANDITS> samples;
    for (std::size_t i = 0; i < N_BANDITS; i++) {
      // if (banditChoice[i]) {
      samples[i] = std::normal_distribution<float>(hiddenBanditMeans[i],
                                                   hiddenBanditStd[i])(engine);
      //} else {
      //  samples[i] = 0; // Only the sampled bandit is seen
      //}
    }
    return samples;
  }

  StateType step(const StateType &state) const override {
    return StateType{state.hiddenRandomEngine, state.hiddenBanditMeans,
                     state.hiddenBanditStd,
                     this->sample(state.hiddenRandomEngine,
                                  state.hiddenBanditMeans,
                                  state.hiddenBanditStd)};
  }

  friend std::ostream &operator<<(std::ostream &os, const BanditAction &b) {
    os << "BanditAction(" << std::get<0>(b) << ")";
    return os;
  }
};

template <std::size_t N_BANDITS>
struct BanditStep : environment::Step<BanditAction<N_BANDITS>> {
  using BaseType = environment::Step<BanditAction<N_BANDITS>>;
  using typename BaseType::ActionSpace;
  using typename BaseType::PrecisionType;
  using typename BaseType::StateType;

  using BaseType::step;
};

template <std::size_t N_BANDITS, environment::RewardType REWARD_T,
          environment::ReturnType RETURN_T>
struct BanditEnvironment
    : environment::Environment<BanditStep<N_BANDITS>, REWARD_T, RETURN_T> {

  SETUP_TYPES(SINGLE_ARG(
      environment::Environment<BanditStep<N_BANDITS>, REWARD_T, RETURN_T>));
  using EnvironmentType = BanditEnvironment<N_BANDITS, REWARD_T, RETURN_T>;

  constexpr static std::size_t N = N_BANDITS;

  std::minstd_rand &engine;
  std::normal_distribution<PrecisionType> prior_mu;
  std::normal_distribution<PrecisionType> prior_var;
  std::array<PrecisionType, N_BANDITS> means;
  std::array<PrecisionType, N_BANDITS> stddevs;

  BanditEnvironment(std::minstd_rand &engine, PrecisionType prior_mu_ave = 0,
                    PrecisionType prior_mu_stddev = 1,
                    PrecisionType prior_var_ave = 0,
                    PrecisionType prior_var_stddev = 1)
      : engine(engine), prior_mu(prior_mu_ave, prior_mu_stddev),
        prior_var(prior_var_ave, prior_var_stddev),
        BaseType(StateType::nullFactory(engine, means, stddevs)) {
    reset();
  }

  void resetDistributions() {

    // calculate means
    for (std::size_t i = 0; i < N_BANDITS; i++) {
      do {
        means[i] = prior_mu(engine);
      } while (means[i] <= 0);
      do {
        stddevs[i] = prior_var(engine);
      } while (stddevs[i] <= 0);
    }
  }

  StateType reset() override {

    resetDistributions();

    // Setup the initial State
    this->state = StateType::nullFactory(engine, means, stddevs);

    return this->state;
  }

  void printDistributions() const {
    for (int i = 0; i < N_BANDITS; i++) {
      std::cout << "bandit " << i << ": " << means[i] << " (+/- " << stddevs[i]
                << ")\n";
    }
  }

  StateType getNullState() const {
    return StateType(engine, means, stddevs, {0});
  }

  std::vector<ActionSpace> getReachableActions(const StateType &s) const {
    std::vector<ActionSpace> actions;
    for (int i = 0; i < N_BANDITS; i++) {
      actions.push_back(
          ActionSpace(typename ActionSpace::BaseType::DataType{i}));
    }
    return actions;
  }
};

} // namespace bandit

namespace bandit::rewards {

template <std::size_t N_BANDITS, float SUCCESS_V = 1.0F, float FAIL_V = 0.0F>
struct ConstantReward : environment::Reward<BanditAction<N_BANDITS>> {

  using BaseType = environment::Reward<BanditAction<N_BANDITS>>;
  using typename BaseType::ActionSpace;
  using typename BaseType::PrecisionType;
  using typename BaseType::TransitionType;

  static constexpr PrecisionType successVal = SUCCESS_V;
  static constexpr PrecisionType failVal = FAIL_V;

  static PrecisionType reward(const TransitionType &transition) {
    PrecisionType reward = 0.0F;

    const auto actions = std::get<0>(transition.action);
    for (const auto &val : actions) {
      reward += transition.nextState
                    .observableBanditSample[static_cast<std::size_t>(val)];
    }

    return reward;
  };
};

} // namespace bandit::rewards
