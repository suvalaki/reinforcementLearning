#pragma once
#include <array>
#include <type_traits>
#include <xtensor/xfixed.hpp>

namespace spec {

template <typename T>
concept Float = std::is_floating_point<T>::value;

template <typename T, float MIN, float MAX, std::size_t... DIMS>
struct BoundedAarraySpec {
  using ValueType = T;
  using Shape = xt::xshape<DIMS...>;
  using DataType = xt::xtensor_fixed<double, Shape>;
  constexpr static Shape shape = Shape{};
  constexpr static T min = MIN;
  constexpr static T max = MAX;
  constexpr static std::array<std::size_t, sizeof...(DIMS)> dims = {DIMS...};
};

template <typename T>
concept BoundedArraySpecType = requires {
  typename T::ValueType;
  typename T::Shape;
  typename T::DataType;
  T::min;
  T::max;
  T::dims;
}
&&(std::is_integral_v<typename T::ValueType> ||
   std::is_floating_point_v<typename T::ValueType>);

template <typename T>
concept BoundedArraySpecProtocol = requires(T t) {
  {t.min};
  {t.max};
  {t.dims};
}
&&(std::is_integral_v<typename T::ValueType> ||
   std::is_floating_point_v<typename T::ValueType>);

template <typename T>
concept isBoundedArraySpec =
    BoundedArraySpecType<T> && BoundedArraySpecProtocol<T>;

template <typename T>
concept EnumType = std::is_enum_v<T>;

template <EnumType CHOICES, std::size_t NCHOICE, std::size_t... DIMS>
struct CategoricalArraySpec {
  using ChoicesType = CHOICES;
  using Shape = xt::xshape<DIMS...>;
  using DataType = xt::xtensor_fixed<double, Shape>;
  constexpr static Shape shape = Shape{};
  constexpr static std::size_t min = 0;
  constexpr static std::size_t max = NCHOICE;
  constexpr static std::array<std::size_t, sizeof...(DIMS)> dims = {DIMS...};
};

template <typename T>
concept CategoricalArraySpecType = requires {
  typename T::ChoicesType;
  typename T::Shape;
  typename T::DataType;
  T::dims;
}
&&EnumType<typename T::ChoicesType>;

template <typename T>
concept CategoricalArraySpecProtocol = requires(T t) {
  {t.dims};
}
&&EnumType<typename T::ChoicesType>;

template <typename T>
concept isCategoricalArraySpec =
    CategoricalArraySpecType<T> && CategoricalArraySpecProtocol<T>;

template <typename T>
concept AnyArraySpecType = isBoundedArraySpec<T> || isCategoricalArraySpec<T>;

template <typename... T>
concept AllElementsAnyArraySpecType = (AnyArraySpecType<T> && ...);

template <AllElementsAnyArraySpecType... T>
struct CompositeArraySpec : std::tuple<T...> {
  using tupleType = std::tuple<T...>;
  using DataType = std::tuple<typename T::DataType...>;
};

template <typename T>
concept hasTupleType = requires {
  typename T::tupleType;
};

// isinstance of CompositeArraySpec only if check all tuple elements are
// AnyArraySpecType
template <typename T>
concept CompositeArraySpecType =
    hasTupleType<T> &&[]<std::size_t... N>(std::index_sequence<N...>) {
  return (AnyArraySpecType<std::tuple_element_t<N, typename T::tupleType>> &&
          ...);
}
(std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());

// Turn spec into a tuple
template <isBoundedArraySpec T> struct BoundedArray {
  using SpecType = T;
  using typename SpecType::DataType;
  using ValueType = typename T::ValueType;
  constexpr static ValueType min = T::min;
  constexpr static ValueType max = T::max;
  DataType array;
};

} // namespace spec
