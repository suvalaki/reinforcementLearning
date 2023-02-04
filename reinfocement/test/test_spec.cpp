#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "policy/random_policy.hpp"
#include "spec.hpp"
#include <xtensor/xfixed.hpp>

using namespace spec;

TEST_CASE("BoundedAarraySpec instantiation", "[spec][BoundedAarraySpec]") {

  using IntegerSpec = spec::BoundedAarraySpec<int, -5.0F, 10.0F, 1>;
  using IntegerSpec1 = spec::BoundedAarraySpec<int, -5.0F, 10.0F, 5, 6>;

  SECTION("Integer type") {

    // is finite because ints with min and max are finite
    auto instanceIntegerData = default_spec_gen<IntegerSpec>();
    static_assert(std::is_same_v<decltype(instanceIntegerData), xt::xtensor_fixed<int, xt::xshape<1>>>);
    CHECK(instanceIntegerData[0] == -5);

    auto instanceIntegerData1 = default_spec_gen<IntegerSpec1>();
    static_assert(std::is_same_v<decltype(instanceIntegerData1), xt::xtensor_fixed<int, xt::xshape<5, 6>>>);
    for (std::size_t i = 0; i < 5; i++) {
      for (std::size_t j = 0; j < 6; j++) {
        CHECK(instanceIntegerData1(i, j) == -5);
      }
    }

    // Check contant generation is the same as the min
    auto zeroInstanceIntegerData = constant_spec_gen<IntegerSpec>(0);
    CHECK_THROWS(constant_spec_gen<IntegerSpec>(-6));
    CHECK_THROWS(constant_spec_gen<IntegerSpec>(11));
    CHECK(zeroInstanceIntegerData[0] == 0);

    auto zeroInstanceIntegerData1 = constant_spec_gen<IntegerSpec1>(0);
    CHECK_THROWS(constant_spec_gen<IntegerSpec1>(-6));
    CHECK_THROWS(constant_spec_gen<IntegerSpec1>(11));
    for (std::size_t i = 0; i < 5; i++) {
      for (std::size_t j = 0; j < 6; j++) {
        CHECK(zeroInstanceIntegerData1.at(i, j) == 0);
      }
    }

    // Check random generation creates integers between min and max
    auto randomInstanceIntegerData = policy::random_spec_gen<IntegerSpec>();
    CHECK(randomInstanceIntegerData[0] >= -5);
    CHECK(randomInstanceIntegerData[0] <= 10);

    auto randomInstanceIntegerData1 = policy::random_spec_gen<IntegerSpec1>();
    for (std::size_t i = 0; i < 5; i++) {
      for (std::size_t j = 0; j < 6; j++) {
        CHECK(randomInstanceIntegerData1.at(i, j) >= -5);
        CHECK(randomInstanceIntegerData1.at(i, j) <= 10);
      }
    }
  }

  SECTION("Floating type") {}
}