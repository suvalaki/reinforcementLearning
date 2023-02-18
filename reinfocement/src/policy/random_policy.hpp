#pragma once
#include <cmath>
#include <exception>
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "policy.hpp"
#include "spec.hpp"

namespace policy {

// A RandomPolicy is a policy that samples from the action space - It doesnt have any internal state, nor
// an objective to maximise. As such it can be considered a DistributionPolicy (in the sense that it is
// sampling from a uniform distribution over the action space) or a ValuePolicy (in the sense that it is
// returning a value of 1 for all actions) .. ALTERNATIVE: ITS NOT AN VALUE POLICY.

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

template <environment::EnvironmentType E> struct RandomPolicy : virtual Policy<E>, PolicyDistributionMixin<E> {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(E));

  // Get a random event over the bounded specification
  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override {
    return this->sampleAction(e, s);
  }
  virtual void update(const EnvironmentType &e, const TransitionType &s){};
  ActionSpace sampleAction(const EnvironmentType &e, const StateType &s) const override;
};

// impl PolicyDistributionMixin - the other methods should be implemented by the user for their specific policy
template <environment::EnvironmentType E>
typename RandomPolicy<E>::ActionSpace RandomPolicy<E>::sampleAction(const EnvironmentType &e,
                                                                    const StateType &s) const {
  return ActionSpace{random_spec_gen<typename ActionSpace::SpecType>()};
}

} // namespace policy