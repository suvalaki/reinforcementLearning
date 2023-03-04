#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "dummy_environment.hpp"
#include "environment.hpp"

namespace policy::objectives {

/** @brief  A generic factory to create the value type (in order to give starting values)
 */
template <typename VALUE_TYPE, typename... FIELDS> struct ValueFactory;
template <typename VALUE_TYPE, typename... FIELDS> struct ValueFactory<VALUE_TYPE, std::tuple<FIELDS...>> {
  using ValueType = VALUE_TYPE;
  using FieldTypes = std::tuple<FIELDS...>;
  using PrecisionType = typename ValueType::PrecisionType;
  ValueType create(const FIELDS &...fields) { return ValueType{fields...}; }
  ValueType create(const FieldTypes &fields) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) { return ValueType{std::get<I>(fields)...}; }
    (std::make_index_sequence<sizeof...(FIELDS)>{});
  }
  virtual void update() {}
  virtual void update(const PrecisionType &reward) {}
};

template <typename T>
concept isValueFactory = requires(T t) {
  { t.create(std::declval<typename T::FieldTypes>()) } -> std::same_as<typename T::ValueType>;
  {t.update()};
};

/** @brief  The base class for all state action values. For a given environment type E, the value
 */
template <environment::EnvironmentType E> struct Value {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(E))
  using FieldTypes = std::tuple<PrecisionType>;
  using Factory = ValueFactory<Value, FieldTypes>;

  PrecisionType value = 0;

  Value(const PrecisionType &value = 0) : value(value) {}
  Value(const FieldTypes &valueTuple) : value(std::get<0>(valueTuple)) {}
  // operator<=>
  bool operator<(const Value &other) const { return value < other.value; }
  bool operator<(const PrecisionType &other) const { return value < other; }
  bool operator>(const PrecisionType &other) const { return value > other; }
  bool operator==(const PrecisionType &other) const { return value == other; }
  virtual bool operator==(const Value &other) const { return value == other.value; }
  virtual void noFocusUpdate() {}
  Value operator+(const Value &other) const { return Value{value + other.value}; }
  Value &operator+=(const Value &other) {
    this->value += other.value;
    return *this;
  }

  PrecisionType getValue() const { return value; }
};

template <typename T>
concept isValue = std::is_base_of_v<Value<typename T::EnvironmentType>, T> && isValueFactory<typename T::Factory>;

template <template <typename> class C>
concept isValueTemplate =
    // using finite here because if this is called from a finite env check, it will be a finite value. C can have checks
    // that need to be finite. For example, the finite value factory needs a finite environment.
    std::is_base_of_v<Value<environment::dummy::DummyFiniteEnvironment>, C<environment::dummy::DummyFiniteEnvironment>>;

} // namespace policy::objectives
