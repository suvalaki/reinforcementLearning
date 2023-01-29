#pragma once
#include <array>
#include <iostream>
#include <string>
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
  constexpr static std::size_t nDim = sizeof...(DIMS);
};

template <typename T>
concept BoundedArraySpecType = requires {
  typename T::ValueType;
  typename T::Shape;
  typename T::DataType;
  T::min;
  T::max;
  T::dims;
  T::nDim;
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
  constexpr static std::size_t nDim = sizeof...(DIMS);
};

template <typename T>
concept CategoricalArraySpecType = requires {
  typename T::ChoicesType;
  typename T::Shape;
  typename T::DataType;
  T::shape;
  T::min;
  T::max;
  T::dims;
  T::nDim;
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

template <typename T> struct type_getter_bound { using type = T::ValueType; };
template <typename T> struct type_getter_cat { using type = double; };

template <AllElementsAnyArraySpecType... T>
struct CompositeArray : std::tuple<typename T::DataType...> {
  using tupleType = std::tuple<T...>;
  using tupleDataType = std::tuple<typename T::DataType...>;
  using tupleValueType = std::tuple<
      typename std::conditional_t<isCategoricalArraySpec<T>, type_getter_cat<T>,
                                  type_getter_bound<T>>::type...>;

  using tupleDataType::tupleDataType;

  friend std::ostream &operator<<(std::ostream &os, const CompositeArray &rhs) {
    os << "CompositeArray(";
    std::apply([&os](auto &&...args) { ((os << args << ", "), ...); },
               static_cast<const tupleDataType &>(rhs));
    os << ")";
    return os;
  }

  std::size_t hash() const {
    auto ss = std::stringstream();
    ss << *this;
    std::hash<std::string> hasher;
    std::size_t h = hasher(ss.str());
    return h;
  }

  friend bool operator==(const CompositeArray &lhs, const CompositeArray &rhs) {

    if constexpr (sizeof...(T) == 0) {
      return true;
    }

    return [&]<std::size_t... N>(std::index_sequence<N...>) {
      // xtensor == comparison returns a bool for the entire thing
      // https://xtensor.readthedocs.io/en/latest/quickref/operator.html
      return ((std::get<N>(lhs) == std::get<N>(rhs)) && ...);
    }
    (std::make_index_sequence<sizeof...(T)>());
  }
};

template <AllElementsAnyArraySpecType... T>
struct CompositeArraySpec : std::tuple<T...> {
  using tupleType = std::tuple<T...>;
  using DataType = CompositeArray<T...>;
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

// methods to return a default data type given the spec
// For now we are using the min as the default

template <isCategoricalArraySpec T> decltype(auto) default_spec_gen() {
  return xt::ones<double>(T::shape);
}

template <isBoundedArraySpec T> decltype(auto) default_spec_gen() {
  return xt::ones<typename T::ValueType>(T::shape);
}

template <CompositeArraySpecType T> typename T::DataType default_spec_gen() {

  // Tuple of the random types
  return []<std::size_t... N>(std::index_sequence<N...>) {
    return typename T::DataType(
        default_spec_gen<std::tuple_element_t<N, typename T::tupleType>>()...);
  }
  (std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());
}

// Mechanism for a constant data for the spec

template <isBoundedArraySpec T>
std::enable_if_t<isBoundedArraySpec<T>, typename T::DataType>
constant_spec_gen(const typename T::ValueType &value) {
  return value * xt::ones<typename T::ValueType>(T::shape);
}

template <isBoundedArraySpec T>
std::enable_if_t<isCategoricalArraySpec<T>, typename T::DataType>
constant_spec_gen(const typename T::ValueType &value) {
  return value * xt::ones<double>(T::shape);
}

// Applies the same constant to all elements of the composite array
template <CompositeArraySpecType T>
requires CompositeArraySpecType<T> std::enable_if_t < CompositeArraySpecType<T>,
typename T::DataType > constant_spec_gen(const double &value) {

  // Tuple of the random types
  return [&value]<std::size_t... N>(std::index_sequence<N...>) {
    return typename T::DataType(
        constant_spec_gen<std::tuple_element_t<N, typename T::tupleType>>(
            static_cast<
                typename std::tuple_element_t<N, typename T::tupleValueType>>(
                value))...);
  }
  (std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());
}

// Specify each element for each eleemnt of the spec
template <CompositeArraySpecType T>
requires CompositeArraySpecType<T> std::enable_if_t < CompositeArraySpecType<T>,
typename T::DataType >
    constant_spec_gen(const typename T::tupleValueType &value) {

  // Tuple of the random types
  return [&value]<std::size_t... N>(std::index_sequence<N...>) {
    return typename T::DataType(
        constant_spec_gen<std::tuple_element_t<N, typename T::tupleType>>(
            static_cast<
                typename std::tuple_element_t<N, typename T::tupleValueType>>(
                value))...);
  }
  (std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());
}

// Turn spec into a tuple
template <isBoundedArraySpec T> struct BoundedArray {
  using SpecType = T;
  using typename SpecType::DataType;
  using ValueType = typename T::ValueType;
  constexpr static ValueType min = T::min;
  constexpr static ValueType max = T::max;
  DataType array = default_spec_gen<SpecType>();
};

} // namespace spec

// extend std::get
namespace std {
template <std::size_t I, spec::AllElementsAnyArraySpecType... T>
decltype(auto) get(spec::CompositeArray<typename T::DataType...> &v) {
  return std::get<I>(
      static_cast<
          spec::CompositeArray<typename T::DataType...>::tupleDataType &>(v));
}
} // namespace std