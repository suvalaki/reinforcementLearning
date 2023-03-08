#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "policy/objectives/value.hpp"
#include "utils/tmp.hpp"

namespace policy::objectives {

/** @brief For a finite environment (one with a finite number of state x actions combinations) the value
 *        of a state action pair is a tuple of the value and the number of times it has been visited.
 */
template <environment::FiniteEnvironmentType E>
struct FiniteValue : Value<E> {

  using ValueType = FiniteValue;
  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(E))
  using BaseType = Value<E>;
  using FieldTypes = tuple_cat_t<typename Value<E>::FieldTypes, std::tuple<std::size_t>>;
  std::size_t step = 1;

  FiniteValue(const PrecisionType &value = 0, const std::size_t &step = 1) : BaseType(value), step(step) {}
  virtual bool operator==(const FiniteValue &other) const;
  FiniteValue operator+(const FiniteValue &other) const;
  FiniteValue &operator+=(const FiniteValue &other);

  struct Factory : ValueFactory<FiniteValue, FieldTypes> {
    using BaseType = ValueFactory<FiniteValue, FieldTypes>;
    using PrecisionType = typename BaseType::PrecisionType;
    PrecisionType averageReturn = 0;
    std::size_t step = 0;
    using BaseType::create;
    using BaseType::update;
    void update(const PrecisionType &reward) override;
  };
};

template <environment::FiniteEnvironmentType E>
auto FiniteValue<E>::operator==(const FiniteValue &other) const -> bool {
  return this->value == other.value && this->step == other.step;
}

template <environment::FiniteEnvironmentType E>
auto FiniteValue<E>::operator+(const FiniteValue &other) const -> FiniteValue {
  return FiniteValue{this->value + other.value};
}

template <environment::FiniteEnvironmentType E>
auto FiniteValue<E>::operator+=(const FiniteValue &other) -> FiniteValue & {
  this->value += other.value;
  return *this;
}

template <environment::FiniteEnvironmentType E>
void FiniteValue<E>::Factory::update(const PrecisionType &reward) {
  averageReturn = (averageReturn * step + reward) / (step + 1);
  step++;
}

template <typename T>
concept isFiniteValue =
    std::is_base_of_v<FiniteValue<typename T::EnvironmentType>, T> && isValueFactory<typename T::Factory>;

template <template <typename> class C>
concept isFiniteValueTemplate = std::is_base_of_v<
    FiniteValue<environment::dummy::DummyFiniteEnvironment>,
    C<environment::dummy::DummyFiniteEnvironment>>;

} // namespace policy::objectives
