#include "catch.hpp"

#include "exceptions.hpp"
#include "strategy.hpp"

#include <random>
#include <type_traits>
#include <vector>

TEST_CASE("strategy::action_choice::ActionChoiceFunctor") {
  // Default virtual functor is implemented to always return the EXPLOIT choice
  { // Default Constructor doesnt need pointers/refs to data
    auto c = strategy::action_choice::ActionChoiceFunctor<float>();
    CHECK(c() == strategy::action_choice::ActionTypes::EXPLOIT);
  }
  { // Secondary constructor enables to use of data
    auto v = std::vector<float>(10, 0.1);
    auto s = std::vector<std::size_t>(10, 1);
    auto c = strategy::action_choice::ActionChoiceFunctor<float>(v, s);
    CHECK(c() == strategy::action_choice::ActionTypes::EXPLOIT);
  }
}

TEST_CASE("strategy::action_choice::ConstantSelectorFunction") {

  // The constant choice functor always returns the action type supplied to it
  // during construction
  {
    // When supplying exploit as the constant action type
    auto c = strategy::action_choice::ConstantSelectionFunctor<float>(
        strategy::action_choice::ActionTypes::EXPLOIT);
    CHECK(c() == strategy::action_choice::ActionTypes::EXPLOIT);
    // double check public polymorphism
    CHECK(dynamic_cast<strategy::action_choice::ActionChoiceFunctor<float> *>(
        &c));
  }
  {
    // When supplying explore as the constant action type
    auto c = strategy::action_choice::ConstantSelectionFunctor<float>(
        strategy::action_choice::ActionTypes::EXPLORE);
    CHECK(c() == strategy::action_choice::ActionTypes::EXPLORE);
    // double check public polymorphism
    CHECK(dynamic_cast<strategy::action_choice::ActionChoiceFunctor<float> *>(
        &c));
  }
}

TEST_CASE("strategy::action_choice::BinarySelectionFunctor") {

  // The epsilon valuie of the Binary selection functor denotes the probablity
  // of exploration
  std::minstd_rand generator = {};
  {
    // Check that when the probability of exploration is set to zero that the
    // function exploits
    auto b =
        strategy::action_choice::BinarySelectionFunctor<float>(0.0, generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLOIT);
    // double check public polymorphism
    CHECK(dynamic_cast<strategy::action_choice::ActionChoiceFunctor<float> *>(
        &b));
  }
  {
    // Check that when the probability of exploration is set to one that the
    // function explores
    auto b =
        strategy::action_choice::BinarySelectionFunctor<float>(1.0, generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLORE);
    // double check public polymorphism
    CHECK(dynamic_cast<strategy::action_choice::ActionChoiceFunctor<float> *>(
        &b));
  }
  // The functor accounts for epsilon values out of the bounds of admissible
  // probabilities by setting them to the nearest bound. When probability is set
  // to less than zero we bound it at zero and when it is greeater than one we
  // bound it by one.
  {
    // Check that when the probability of exploration is set to less than
    // one that the constructor throws
    CHECK_THROWS(strategy::action_choice::BinarySelectionFunctor<float>(
        -0.1F, generator));
  }
  {
    // Check that when the probability of exploration is set to grerater than
    // one that the constructor throws
    CHECK_THROWS(strategy::action_choice::BinarySelectionFunctor<float>(
        1.1F, generator));
  }
}

TEST_CASE("strategy::explore::ExploreFunctor") {
  // Explore functor default virtual operator method always returns 0
  auto v = std::vector<float>(10, 0.1);
  auto s = std::vector<std::size_t>(10, 1);
  auto functor = strategy::explore::ExploreFunctor(v, s);
  CHECK(functor() == 0);
}

TEST_CASE("strategy::explore::RandomActionSelectionFunctor") {
  // Random Selection functor selects a random number within the size of the
  // index for the valueEstimateVector.
  // We safely assume that the fector has at least size 1
  { // When the size of the valueEstimate vector is at a minimum
    auto v = std::vector<float>(1, 0.1);
    auto s = std::vector<std::size_t>(1, 1);
    std::minstd_rand generator = {};
    auto functor =
        strategy::explore::RandomActionSelectionFunctor(v, s, generator);
    std::minstd_rand generator_copy = generator; // copy construction
    std::uniform_int_distribution<std::size_t> distribution_benchmark(0, 1);

    // Check that the distribution and the select functor yield the same as one
    // another. We assume that <random> is well tested and is yielding correct
    // random integers.
    for (std::size_t i = 0; i < 100; i++) {
      auto functorChoice = functor();
      auto benchmakChoice = distribution_benchmark(generator_copy);
      CHECK(functorChoice == benchmakChoice);
    }
    // double check public polymorphism
    CHECK(dynamic_cast<strategy::explore::ExploreFunctor<float> *>(&functor));
  }
  { // When the size of the valueEstimate vector is some random int
    auto v = std::vector<float>(10, 0.1);
    auto s = std::vector<std::size_t>(10, 1);
    std::minstd_rand generator = {};
    auto functor =
        strategy::explore::RandomActionSelectionFunctor<float>(v, s, generator);
    std::minstd_rand generator_copy = generator; // copy construction
    std::uniform_int_distribution<std::size_t> distribution_benchmark(0, 10);

    // Check that the distribution and the select functor yield the same as one
    // another. We assume that <random> is well tested and is yielding correct
    // random integers.
    for (std::size_t i = 0; i < 100; i++) {
      auto functorChoice = functor();
      auto benchmakChoice = distribution_benchmark(generator_copy);
      CHECK(functorChoice == benchmakChoice);
    }
    // double check public polymorphism
    CHECK(dynamic_cast<strategy::explore::ExploreFunctor<float> *>(&functor));
  }
}

TEST_CASE("strategy::step_size::StepSizeFunctor") {
  auto s = std::vector<std::size_t>(10, 1);
  auto functor = strategy::step_size::StepSizeFunctor<float>(s);
  for (std::size_t action = 0; action < 10; action++) {
    // Check that for each action we get the same
    CHECK(functor(action) == 0);
  }
  // It must be noted that the StepSize functor doesnt have any guards
  // preventing negative step sizes. In practicce it doesnt make sense to have
  // negative steps (they will never converge)
}

TEST_CASE("strategy::step_size::ConstantStepSizeFunctor") {

  int maxVal = 10;
  auto s = std::vector<std::size_t>(10, 1);
  {
    // Check over a range of float values that the constant step size is adhered
    // to
    for (int stepSize = 0; stepSize < maxVal; stepSize++) {
      auto functor = strategy::step_size::ConstantStepSizeFunctor<float>(
          s, static_cast<float>(stepSize));
      for (std::size_t action = 0; action < 10; action++) {
        // Check that for each action we get the same
        CHECK(functor(action) == static_cast<float>(stepSize));
      }
      // double check public polymorphism to StepSizeFunctor
      CHECK(dynamic_cast<strategy::step_size::StepSizeFunctor<float> *>(
          &functor));
    }
  }
}

TEST_CASE("strategy::step_size::SampleAverageStepSizeFunctor") {

  {
    std::size_t nElements = 10;
    std::size_t maxCount = 50;
    auto s = std::vector<std::size_t>(nElements, 1);
    // Check over increasing counts (until maxCount is reached ) for each
    // element within s (the sample size counts)
    for (int elementCount = 0; elementCount < maxCount; elementCount++) {
      auto functor =
          strategy::step_size::SampleAverageStepSizeFunctor<float>(s);
      for (std::size_t action = 0; action < nElements; action++) {
        s[action] = elementCount;
        // Check that for each action we get the same
        CHECK(functor(action) == 1.0F / static_cast<float>(elementCount));
      }
      // double check public polymorphism to StepSizeFunctor
      CHECK(dynamic_cast<strategy::step_size::StepSizeFunctor<float> *>(
          &functor));
    }
  }
}

TEST_CASE("strategy::exploit::argmax") {
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    CHECK(strategy::exploit::argmax(v) == 4);
  }
  {
    std::vector<float> v = {0, 0, 0, 0};
    CHECK(strategy::exploit::argmax(v) == 0);
  }
}

TEST_CASE("strategy::exploit::upperConfidenceBoundActionSelection") {
  { //  Test zero case
    std::vector<float> v = {1, 2, 3, 4, 5};
    std::vector<std::size_t> s = {0, 0, 0, 0, 0};
    CHECK(strategy::exploit::upperConfidenceBoundActionSelection(1.96F, v, s) ==
          4);
  }
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    std::vector<std::size_t> s = {1, 2, 3, 4, 5};
    // index 0: 1 + 1.96 * sqrt( ln(15) / 1 ) ~= 4.225
    // index 1: 2 + 1.96 * sqrt( ln(15) / 2 ) ~= 4.28
    // index 2: 3 + 1.96 * sqrt( ln(15) / 3 ) ~= 4.865
    // index 3: 4 + 1.96 * sqrt( ln(15) / 4 ) ~= 5.61
    // index 4: 5 + 1.96 * sqrt( ln(15) / 5 ) ~= 7.31
    // index 4 is the maximum
    CHECK(strategy::exploit::upperConfidenceBoundActionSelection(1.96F, v, s) ==
          4);
  }
}

TEST_CASE("strategy::exploit::ExploitFunctor") {
  {
    // Default Constructor
    auto functor = strategy::exploit::ExploitFunctor<float>();
    CHECK(functor() == 0);
  }
  {
    // Value Estimate Only Constructor
    std::vector<float> v = {1, 2, 3, 4, 5};
    auto functor = strategy::exploit::ExploitFunctor<float>(v);
    CHECK(functor() == 0);
  }
  {
    // Complete Constructor
    std::vector<float> v = {1, 2, 3, 4, 5};
    std::vector<std::size_t> s = {1, 2, 3, 4, 5};
    auto functor = strategy::exploit::ExploitFunctor<float>(v, s);
    CHECK(functor() == 0);
  }
}

TEST_CASE("strategy::exploit::ArgmaxFunctor") {
  {
    std::vector<float> v = {0, 0, 0, 0};
    std::vector<std::size_t> s = {0, 0, 0, 0};
    strategy::exploit::ArgmaxFunctor<float> aF(v);
    CHECK(aF() == 0);
  }
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    std::vector<std::size_t> s = {0, 0, 0, 0, 0};
    strategy::exploit::ArgmaxFunctor<float> aF = {v};
    CHECK(aF() == 4);
  }
  {
    std::vector<float> v = {1, 2, 10, 4, 5};
    std::vector<std::size_t> s = {0, 0, 0, 0, 0};
    strategy::exploit::ArgmaxFunctor<float> aF = {v};
    CHECK(aF() == 2);
  }
}

TEST_CASE("strategy::exploit::UpperConfidenceBoundFunctor") {
  // Check decomposition to argmax
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    std::vector<std::size_t> s = {0, 0, 0, 0, 0};
    strategy::exploit::UpperConfidenceBoundFunctor<float> functor = {v, s,
                                                                     1.0F};
    CHECK(functor() == 4);
  }
  {
    std::vector<float> v = {1, 2, 10, 4, 5};
    std::vector<std::size_t> s = {5, 4, 3, 2, 1};
    strategy::exploit::UpperConfidenceBoundFunctor<float> functor = {v, s,
                                                                     1.0F};
    CHECK(functor() == 2);
  }
}

TEST_CASE("strategy::StrategyBase") {}

TEST_CASE("strategy::Strategy") {}

TEST_CASE("strategy::GreedyStrategy") {
  {
    std::vector<float> v = {0, 0, 0, 0};
    strategy::GreedyStrategy<float> strat(v);
    CHECK(strat.getActionValueEstimate(0) == 0);
    CHECK(strat.getActionValueEstimate().size() == 4);
    CHECK(strat.getActionSelectionCount().size() == 4);
    CHECK(strat.exploit() == 0);
  }
}

TEST_CASE("strategy::EpsilonGreedyStrategy") {}