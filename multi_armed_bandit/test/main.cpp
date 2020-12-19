#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file
#include "catch.hpp"

unsigned int Factorial(unsigned int number) {
  return number <= 1 ? number : Factorial(number - 1) * number;
}

TEST_CASE("Factorials are computed", "[factorial]") {
  REQUIRE(Factorial(1) == 1);
  REQUIRE(Factorial(2) == 2);
  REQUIRE(Factorial(3) == 6);
  REQUIRE(Factorial(10) == 3628800);
}

#include "strategy.hpp"
#include <vector>

TEST_CASE("Argmax") {
  {
    std::vector<float> v = {1, 2, 3, 4, 5};
    CHECK(strategy::exploit::argmax(v) == 4);
  }
  {
    std::vector<float> v = {0, 0, 0, 0};
    CHECK(strategy::exploit::argmax(v) == 0);
  }
}

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