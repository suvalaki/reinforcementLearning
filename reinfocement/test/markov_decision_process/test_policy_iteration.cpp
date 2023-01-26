#include "catch.hpp"
#include <iostream>

#include "coin_mdp.hpp"
#include "markov_decision_process/finite_state_value_function.hpp"
#include "markov_decision_process/finite_transition_model.hpp"
#include "markov_decision_process/policy_iteration.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/random_policy.hpp"

using namespace environment;
using namespace markov_decision_process;

TEST_CASE("Coin MPD can undergo policy iteration") {

  auto s0 = CoinState{0.0F, {}};
  auto s1 = CoinState{1.0F, {}};
  auto a0 = CoinAction{0};
  auto a1 = CoinAction{1};
  auto transitionModel = CoinTransitionModel{                       //
                                             {T{s0, a0, s0}, 0.8F}, //
                                             {T{s0, a0, s1}, 0.2F}, //
                                             {T{s0, a1, s0}, 0.3F}, //
                                             {T{s0, a1, s1}, 0.7F}, //
                                             {T{s1, a0, s0}, 0.1F}, //
                                             {T{s1, a0, s1}, 0.9F}, //
                                             {T{s1, a1, s0}, 0.5F}, //
                                             {T{s1, a1, s1}, 0.5F}};

  auto environ = CoinEnviron{transitionModel, s0};

  using CoinDistributionPolicy = policy::DistributionPolicy<CoinEnviron>;
  auto policy = CoinDistributionPolicy{};
  // itialise the q-table inside the policy by using the random policy
  policy.initialise(environ, 100);

  policy.printQTable();

  using CoinValueFunction = FiniteStateValueFunction<CoinEnviron, 0.0F, 0.5F>;
  auto valueFunction = CoinValueFunction{};

  SECTION("Several iterations of value iteration steps succesfully update the "
          "value") {
    auto initialValue = valueFunction.valueAt(s0);
    for (int i = 0; i < 100; ++i) {
      auto val = policy_evaluation_step(valueFunction, environ, policy, s0);
      // std::cout << val << "\n";
      valueFunction.valueEstimates.at(s0) = val;
      // valueFunction.policy_improvement_step(environ, policy, s0);
    }

    // Validate that the value iteration has indeed updated the value
    CHECK_FALSE(initialValue == valueFunction.valueEstimates.at(s0));
  }

  SECTION("Complete pass over all states works") {
    // because we need to compare the values we must at least initialise them.
    valueFunction.initialize(environ);
    auto initialValues = valueFunction.valueEstimates;
    // valueFunction.policy_evaluation(environ, policy, 1e-3F);
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    for (auto &[state, value] : valueFunction.valueEstimates) {
      CHECK(value != initialValues.at(state));
    }
  }
}

TEST_CASE("Coin MPD can undergo policy improvement") {

  auto s0 = CoinState{0.0F, {}};
  auto s1 = CoinState{1.0F, {}};
  auto a0 = CoinAction{0};
  auto a1 = CoinAction{1};
  auto transitionModel = CoinTransitionModel{                       //
                                             {T{s0, a0, s0}, 0.8F}, //
                                             {T{s0, a0, s1}, 0.2F}, //
                                             {T{s0, a1, s0}, 0.3F}, //
                                             {T{s0, a1, s1}, 0.7F}, //
                                             {T{s1, a0, s0}, 0.1F}, //
                                             {T{s1, a0, s1}, 0.9F}, //
                                             {T{s1, a1, s0}, 0.5F}, //
                                             {T{s1, a1, s1}, 0.5F}};

  auto environ = CoinEnviron{transitionModel, s0};

  using CoinDistributionPolicy = policy::DistributionPolicy<CoinEnviron>;
  auto policy = CoinDistributionPolicy{};
  // itialise the q-table inside the policy by using the random policy
  policy.initialise(environ, 100);

  policy.printQTable();

  using CoinValueFunction = FiniteStateValueFunction<CoinEnviron, 0.0F, 0.5F>;
  auto valueFunction = CoinValueFunction{};

  SECTION("Policy improvement step updates a policy") {
    // force the q_table to have non-optimal values
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a0)) = 1.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a1)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a0)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a1)) = 0.0F;
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    auto updated = policy_improvement_step(valueFunction, environ, policy, s0);
    CHECK_FALSE(updated); // the policy updated
  }

  SECTION("Complete Policy improvement") {
    // force the q_table to have non-optimal values
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a0)) = 1.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a1)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a0)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a1)) = 0.0F;
    policy_improvement(valueFunction, environ, policy, 1e-3F);
    auto p = policy.getProbability(
        s0, CoinDistributionPolicy::KeyMaker::make(s0, a0));
    CHECK(p != 1.0F);
  }
}