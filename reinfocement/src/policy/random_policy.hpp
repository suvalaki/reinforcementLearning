#pragma once
#include "environment.hpp"
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "policy.hpp"
#include "spec.hpp"

namespace policy {

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

template <isBoundedArraySpec T, class E = xt::random::default_engine_type>
std::enable_if_t<isBoundedArraySpec<T>, typename T::DataType>
random_spec_gen(E &engine = xt::random::get_default_random_engine()) {

  if constexpr (std::is_integral_v<typename T::ValueType>)
    return xt::random::randint(T::shape, T::min, T::max, engine);

  else if constexpr (std::is_floating_point_v<typename T::ValueType>)
    return xt::random::rand(T::shape, T::min, T::max, engine);

  else
    return xt::zeros(T::shape);
};

template <isCategoricalArraySpec T, class E = xt::random::default_engine_type>
std::enable_if_t<isCategoricalArraySpec<T>, typename T::DataType>
random_spec_gen(E &engine = xt::random::get_default_random_engine()) {
  return xt::random::randint(T::shape, T::min, T::max, engine);
};

template <CompositeArraySpecType T, class E = xt::random::default_engine_type>
requires CompositeArraySpecType<T> std::enable_if_t < CompositeArraySpecType<T>,
typename T::DataType > random_spec_gen(E &engine = xt::random::get_default_random_engine()) {

  // Tuple of the random types
  return [&engine]<std::size_t... N>(std::index_sequence<N...>) {
    return typename T::DataType(random_spec_gen<std::tuple_element_t<N, typename T::tupleType>>(engine)...);
  }
  (std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());
}

template <environment::EnvironmentType ENVIRON_T> struct RandomPolicy : Policy<ENVIRON_T> {

  SETUP_TYPES(SINGLE_ARG(Policy<ENVIRON_T>));

  // Get a random event over the bounded specification
  ActionSpace operator()(const StateType &s) override {
    return ActionSpace{random_spec_gen<typename ActionSpace::SpecType>()};
  }

  virtual void update(const TransitionType &s){};
};

template <environment::FiniteEnvironmentType ENVIRON_T,
          isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRON_T>>
struct FiniteRandomPolicy : FiniteStatePolicy<ENVIRON_T, KEYMAPPER_T> {

  SETUP_TYPES(SINGLE_ARG(FiniteStatePolicy<ENVIRON_T, KEYMAPPER_T>));

  // Get a random event over the bounded specification
  ActionSpace operator()(const StateType &s) override {
    return ActionSpace{random_spec_gen<typename ActionSpace::SpecType>()};
  }

  virtual void update(const TransitionType &s){};
};

} // namespace policy