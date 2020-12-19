#include "strategy.hpp"
#include "catch.hpp"

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

TEST_CASE("argmax") {}

TEST_CASE("upperConfidenceBoundActionSelection") {}

TEST_CASE("ExploitFunctor") {}

TEST_CASE("ArgmaxFunctor") {}

TEST_CASE("UpperConfidenceBoundFunctor") {}

TEST_CASE("StrategyBase") {}

TEST_CASE("Strategy") {}

TEST_CASE("GreedyStrategy") {}

TEST_CASE("EpsilonGreedyStrategy") {}