#include "catch.hpp"

#include "strategy.hpp"

#include <random>
#include <vector>

TEST_CASE("ActionChoiceFunctor") {}

TEST_CASE("ConstantSelectorFunction") {}

TEST_CASE("BinarySelectionFunctor") {

  {
    auto v = std::vector<float>(10, 0.1);
    auto s = std::vector<std::size_t>(10, 1);
    std::minstd_rand generator = {};
    auto b = strategy::action_choice::BinarySelectionFunctor<float>(v, s, 0.0,
                                                                    generator);
    CHECK(b() == strategy::action_choice::ActionTypes::EXPLOIT);
  }
  {
    auto v = std::vector<float>(10, 0.1);
    auto s = std::vector<std::size_t>(10, 1);
    std::minstd_rand generator = {};
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