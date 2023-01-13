#pragma once
#include "action.hpp"
#include "environment.hpp"
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>
#include <random>

namespace policy {

template <environment::EnvironmentType ENVIRON_T> struct Policy {
  using EnvironmentType = ENVIRON_T;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using StateType = typename EnvironmentType::StateType;
  using ActionSpace = typename EnvironmentType::ActionSpace;
  using TransitionType = typename EnvironmentType::TransitionType;

  // Run the policy over the current state of the environment
  virtual ActionSpace operator()(const StateType &s) = 0;
  virtual void update(const TransitionType &s) = 0;
};


template <action::isBoundedArraySpec T, class E = xt::random::default_engine_type > 
 std::enable_if_t<action::isBoundedArraySpec<T>, typename T::DataType> 
 random_spec_gen (E &engine = xt::random::get_default_random_engine()){

   if constexpr (std::is_integral_v<typename T::ValueType>)
      return xt::random::randint(T::shape, T::min, T::max, engine);

   else if constexpr (std::is_floating_point_v<typename T::ValueType>)
      return xt::random::rand(T::shape, T::min, T::max, engine );

   else 
      return xt::zeros(T::shape);
};

template <action::isCategoricalArraySpec T, class E = xt::random::default_engine_type>
 std::enable_if_t<action::isCategoricalArraySpec<T>, typename T::DataType> 
 random_spec_gen (E &engine = xt::random::get_default_random_engine()){

    return xt::random::randint(T::shape, T::min, T::max, engine );

};

template <action::CompositeArraySpecType T, class E = xt::random::default_engine_type> 
requires action::CompositeArraySpecType<T>
 std::enable_if_t<action::CompositeArraySpecType<T>, typename T::DataType> 
 random_spec_gen (E &engine = xt::random::get_default_random_engine()) {

  // Tuple of the random types
  return 
    [&engine]<std::size_t... N>(std::index_sequence<N...>) { 
        return std::make_tuple(
            random_spec_gen<std::tuple_element_t<N, typename T::tupleType>>(engine) ... ); 
      }(std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());

}


template <environment::EnvironmentType ENVIRON_T> struct RandomPolicy : Policy<ENVIRON_T> {
  using baseType = Policy<ENVIRON_T>;
  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType =  typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using TransitionType = typename baseType::TransitionType;


  // Get a random event over the bounded specification 
  ActionSpace operator()(const StateType &s) override {
    return ActionSpace{random_spec_gen<typename ActionSpace::SpecType>() };
  } 

  virtual void update(const TransitionType &s) {};
};

} // namespace policy
