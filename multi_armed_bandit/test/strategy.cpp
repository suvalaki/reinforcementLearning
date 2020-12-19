#include "catch.hpp"

#include "strategy.hpp"

#include <random>
#include <vector>

TEST_CASE("ActionChoiceFunctor") {}

TEST_CASE("strategy::action_choice::ConstantSelectorFunction") {

  // The constant choice functor always returns the action type supplied to it
  // during construction
  auto v = std::vector<float>(10, 0.1);
  auto s = std::vector<std::size_t>(10, 1);
  {
    // When supplying exploit as the constant action type
    auto c = strategy::action_choice::ConstantSelectionFunctor<float>(
        v, s, strategy::action_choice::ActionTypes::EXPLOIT);
    CHECK(c() == strategy::action_choice::ActionTypes::EXPLOIT);
  }
  {
    // When supplying explore as the constant action type
    auto c = strategy::action_choice::ConstantSelectionFunctor<float>(
        v, s, strategy::action_choice::ActionTypes::EXPLORE);
    CHECK(c() == strategy::action_choice::ActionTypes::EXPLORE);
  }
}

TEST_CASE("strategy::action_choice::BinarySelectionFunctor") {

  // The epsilon valuie of the Binary selection functor denotes the probablity
  // of exploration
  auto v = std::vector<float>(10, 0.1);
  auto s = std::vector<std::size_t>(10, 1);
  std::minstd_rand generator = {};
  {
    // Check that when the probability of exploration is set to zero that the
    // function exploits
    auto b = strategy::action_choice::BinarySelectionFunctor<float>(v, s, 0.0,
                                                                    generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLOIT);
  }
  {
    // Check that when the probability of exploration is set to one that the
    // function explores
    auto b = strategy::action_choice::BinarySelectionFunctor<float>(v, s, 1.0,
                                                                    generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLORE);
  }
  // The functor accounts for epsilon values out of the bounds of admissible
  // probabilities by setting them to the nearest bound. When probability is set
  // to less than zero we bound it at zero and when it is greeater than one we
  // bound it by one.
  {
    // Check that when the probability of exploration is set to less than zero
    // that the function only exploits
    auto b = strategy::action_choice::BinarySelectionFunctor<float>(v, s, -1.0,
                                                                    generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLOIT);
  }
  {
    // Check that when the probability of exploration is set to grerater than
    // one that the function only explores
    auto b = strategy::action_choice::BinarySelectionFunctor<float>(v, s, 1.0,
                                                                    generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLORE);
  }
}

TEST_CASE("ExploreFunctor") {}

TEST_CASE("RandomActionSelectionFunctor") {}

TEST_CASE("StepSizeFunctor") {}

TEST_CASE("ConstantStepSizeFunctor") {}

TEST_CASE("SampleAverageStepSizeFunctor") {}

TEST_CASE("argmax") {
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    CHECK(strategy::exploit::argmax(v) == 4);
  }
  {
    std::vector<float> v = {0, 0, 0, 0};
    CHECK(strategy::exploit::argmax(v) == 0);
  }
}

TEST_CASE("upperConfidenceBoundActionSelection") {}

TEST_CASE("ExploitFunctor") {}

TEST_CASE("ArgmaxFunctor") {
  {
    std::vector<float> v = {0, 0, 0, 0};
    std::vector<std::size_t> s = {0, 0, 0, 0};
    strategy::exploit::ArgmaxFunctor<float> aF = {v, s};
    CHECK(aF() == 0);
  }
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    std::vector<std::size_t> s = {0, 0, 0, 0, 0};
    strategy::exploit::ArgmaxFunctor<float> aF = {v, s};
    CHECK(aF() == 4);
  }
  {
    std::vector<float> v = {1, 2, 10, 4, 5};
    std::vector<std::size_t> s = {0, 0, 0, 0, 0};
    strategy::exploit::ArgmaxFunctor<float> aF = {v, s};
    CHECK(aF() == 2);
  }
}

TEST_CASE("UpperConfidenceBoundFunctor") {}

TEST_CASE("StrategyBase") {}

TEST_CASE("Strategy") {}

TEST_CASE("GreedyStrategy") {
  {
    std::vector<float> v = {0, 0, 0, 0};
    strategy::GreedyStrategy<float> strat(v);
    CHECK(strat.getActionValueEstimate(0) == 0);
    CHECK(strat.getActionValueEstimate().size() == 4);
    CHECK(strat.getActionSelectionCount().size() == 4);
    CHECK(strat.exploit() == 0);
  }
}

TEST_CASE("EpsilonGreedyStrategy") {}