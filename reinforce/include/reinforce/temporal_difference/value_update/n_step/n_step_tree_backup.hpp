#pragma once

/**
 * The treebackup algorithm can be written as the following relation:
 *
 *    G_{t:t+n} = Q(S_t, A_t) + \sum_{k=t}^{\min(t+n-1, T-1)} \delta_k \prod_{i=t+1}^{k} \gamma \pi(A_i | S_i)
 *
 *    where
 *    \delta_t = R_{t+1} + \gamma * V_t (S_{t+1}) - Q(S_t, A_t)
 *    V_t(s) = \sum_{a} \pi(a|s) Q_t(s, a)
 *
 *    Basically take the policy value for all leaf nodes where we do not continue the tree
 *    It is the sum of the expectaion-based TD errors.
 *
 */

namespace temporal_difference {

template <typename CRTP>
struct NStepTreeBackupReturn {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);
  using ExpandedStatefulUpdateResult = typename CRTP::ExpandedStatefulUpdateResult;

  struct ReturnMetrics {
    const PrecisionType ret = 0;
  };

  ReturnMetrics calculateReturn(
      ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator start,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator end) {

    // t+1 is always positioned at the end of the iterator.
    // we provide tau as the start.

    const auto last = end - 1;
    const auto rstart = std::make_reverse_iterator(start);
    const auto rend = std::make_reverse_iterator(end);

    auto G = [&]() {
      if ( last->isDone)
        return last->reward;

      const auto availableActions = environment.getReachableActions(last->transition.state);
      return last->reward +
             discountRate *
                 std::accumulate(
                     availableActions.begin(), availableActions.end(), 0.0F, [&](const auto &v, const auto &a) {
                       return v + policy.getProbability(environment, last->transition.state, a) *
                                      valueFunction.valueAt(KeyMaker::make(environment,last->transition.state, a));
                     });
    }();

    G = std::accumulate(rend, rstart + 1, 0.0F, [&, currentDiscountRate = 1.0F](const auto &acc, const auto &itr) mutable {
      currentDiscountRate *= discountRate;

      // TODO check if contains start element

      if (itr.isDone)
        return itr.reward;

      const auto availableActions = environment.getReachableActions(itr.transition.state);
      return acc

             + itr.reward

             + currentDiscountRate *
                   std::accumulate(
                       availableActions.begin(),
                       availableActions.end(),
                       0.0F,
                       [&](const auto &v, const auto &a) {
                         if (itr.transition.action == a)
                           return v;
                         return v + policy.getProbability(environment, itr.transition.state, a) *
                                        valueFunction.valueAt(KeyMaker::make(environment, itr.transition.state, a));
                       })

             +
             currentDiscountRate * policy.getProbability(environment, itr.transition.state, itr.transition.action) * G;
    });

    return {G};
  }
};

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using NStepTreeSarsaUpdater =
    NStepUpdater<V, SarsaStepMixin, DefaultStorageInterface, NStepTreeBackupReturn, DefaultNStepValueUpdater>;

} // namespace temporal_difference
