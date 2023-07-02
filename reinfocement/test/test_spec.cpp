#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <tuple>

#include "policy/random_policy.hpp"
#include "spec.hpp"
#include <xtensor/xfixed.hpp>

using namespace spec;

TEST_CASE("BoundedAarraySpec instantiation", "[spec][BoundedAarraySpec]") {

  using typesToCheck = std::tuple<int, float, double>;

  auto testType_impl = [&]<std::size_t k>() {
    using T = std::tuple_element_t<k, typesToCheck>;
    constexpr auto min = static_cast<T>(-5);
    constexpr auto max = static_cast<T>(10);
    using IntegerSpec = spec::BoundedAarraySpec<T, min, max, 1>;
    using IntegerSpec1 = spec::BoundedAarraySpec<T, min, max, 5, 6>;
    using IntegerSpec2 = spec::BoundedAarraySpec<T, min, max, 5, 6>;

    SECTION((std::ostringstream() << "Type: " << typeid(T).name()).str()) {

      // is finite because ints with min and max are finite
      if constexpr (std::is_integral_v<T>) {
        static_assert(IntegerSpec::isFinite);
        static_assert(IntegerSpec1::isFinite);
        static_assert(IntegerSpec2::isFinite);
      } else {
        static_assert(!IntegerSpec::isFinite);
        static_assert(!IntegerSpec1::isFinite);
        static_assert(!IntegerSpec2::isFinite);
      }

      // Check min and max
      static_assert(IntegerSpec::min == min);
      static_assert(IntegerSpec::max == max);
      static_assert(IntegerSpec1::min == min);
      static_assert(IntegerSpec1::max == max);
      static_assert(IntegerSpec2::min == min);
      static_assert(IntegerSpec2::max == max);

      // Check dims
      static_assert(IntegerSpec::dims[0] == 1);
      static_assert(IntegerSpec1::dims[0] == 5);
      static_assert(IntegerSpec1::dims[1] == 6);
      static_assert(IntegerSpec2::dims[0] == 5);
      static_assert(IntegerSpec2::dims[1] == 6);

      // Check nDim
      static_assert(IntegerSpec::nDim == 1);
      static_assert(IntegerSpec1::nDim == 2);
      static_assert(IntegerSpec2::nDim == 2);

      // Check shape
      static_assert(IntegerSpec::shape[0] == 1);
      static_assert(IntegerSpec1::shape[0] == 5);
      static_assert(IntegerSpec1::shape[1] == 6);
      static_assert(IntegerSpec2::shape[0] == 5);
      static_assert(IntegerSpec2::shape[1] == 6);

      // Check value type
      static_assert(std::is_same_v<typename IntegerSpec::ValueType, T>);
      static_assert(std::is_same_v<typename IntegerSpec1::ValueType, T>);
      static_assert(std::is_same_v<typename IntegerSpec2::ValueType, T>);

      // Check data type
      static_assert(std::is_same_v<typename IntegerSpec::DataType, xt::xtensor_fixed<T, xt::xshape<1>>>);

      using IntegerSpec = spec::BoundedAarraySpec<T, min, max, 1>;
      using IntegerSpec1 = spec::BoundedAarraySpec<T, min, max, 5, 6>;
      using IntegerSpec2 = spec::BoundedAarraySpec<T, min, max, 5, 6>;

      // is finite because ints with min and max are finite
      auto instanceIntegerData = default_spec_gen<IntegerSpec>();
      static_assert(std::is_same_v<decltype(instanceIntegerData), xt::xtensor_fixed<T, xt::xshape<1>>>);
      CHECK(instanceIntegerData[0] == min);

      auto instanceIntegerData1 = default_spec_gen<IntegerSpec1>();
      static_assert(std::is_same_v<decltype(instanceIntegerData1), xt::xtensor_fixed<T, xt::xshape<5, 6>>>);
      for (std::size_t i = 0; i < 5; i++) {
        for (std::size_t j = 0; j < 6; j++) {
          CHECK(instanceIntegerData1(i, j) == -5);
        }
      }

      // Check contant generation is the same as the min
      auto zeroInstanceIntegerData = constant_spec_gen<IntegerSpec>(0);
      CHECK_THROWS(constant_spec_gen<IntegerSpec>(min - 1));
      CHECK_THROWS(constant_spec_gen<IntegerSpec>(max + 1));
      CHECK(zeroInstanceIntegerData[0] == 0);

      auto zeroInstanceIntegerData1 = constant_spec_gen<IntegerSpec1>(0);
      CHECK_THROWS(constant_spec_gen<IntegerSpec1>(min - 1));
      CHECK_THROWS(constant_spec_gen<IntegerSpec1>(max + 1));
      for (std::size_t i = 0; i < 5; i++) {
        for (std::size_t j = 0; j < 6; j++) {
          CHECK(zeroInstanceIntegerData1.at(i, j) == 0);
        }
      }

      // Check random generation creates integers between min and max
      auto randomInstanceIntegerData = policy::random_spec_gen<IntegerSpec>();
      CHECK(randomInstanceIntegerData[0] >= min);
      CHECK(randomInstanceIntegerData[0] <= max);

      auto randomInstanceIntegerData1 = policy::random_spec_gen<IntegerSpec1>();
      for (std::size_t i = 0; i < 5; i++) {
        for (std::size_t j = 0; j < 6; j++) {
          CHECK(randomInstanceIntegerData1.at(i, j) >= min);
          CHECK(randomInstanceIntegerData1.at(i, j) <= max);
        }
      }
    }
  };

  [&]<std::size_t... k>(std::index_sequence<k...>) {
    (testType_impl.template operator()<k>(), ...);
  }(std::make_index_sequence<std::tuple_size_v<typesToCheck>>{});
}