#include "action.hpp"
#include "bandit.hpp"
#include "bandit_environment.hpp"
#include "bandit_policy.hpp"
#include "policy.hpp"
#include <ctime>
#include <random>

#include "xtensor/xio.hpp"
#include <xtensor/xrandom.hpp>
// XTensor View:
#include "xtensor/xview.hpp"

#include "spec.hpp"

using namespace environment;

// Examples
struct ExampleState : State<float> {
  PrecisionType val;
};

using ExampleSpec0 = spec::BoundedAarraySpec<float, -1.0F, 1.0F, 1>;
using EnumSpec0 =
    spec::CategoricalArraySpec<bandit::BanditActionChoices, 4, 10>;

auto &engine = xt::random::get_default_random_engine();

auto s = 123;
auto z = policy::random_spec_gen<ExampleSpec0>();
auto z1 = policy::random_spec_gen<EnumSpec0>();
auto z2 = policy::random_spec_gen<EnumSpec0>();
auto z3 = policy::random_spec_gen<EnumSpec0>();

static_assert(spec::isBoundedArraySpec<ExampleSpec0>);

using ExampleCombinedSpec = spec::CompositeArraySpec<EnumSpec0>;

// using banditAction = bandit::BanditActionSpec<10>;

struct ExampleAction : Action<ExampleState, ExampleCombinedSpec> {
  using Action<ExampleState, ExampleCombinedSpec>::Action;
};

using ExampleStep = Step<ExampleAction>;
using ExampleTransition = Transition<ExampleAction>;
using ExampleSequence = TransitionSequence<10, ExampleAction>;
struct ExampleReward : Reward<ExampleAction> {
  static PrecisionType reward(const TransitionType &t) { return 10.0F; }
};
using ExampleReturn = Return<ExampleReward>;

using constatntR = bandit::rewards::ConstantReward<10>;
using constatntRet = environment::Return<bandit::rewards::ConstantReward<10>>;
using NBanditEnvironment = bandit::BanditEnvironment<
    10, bandit::rewards::ConstantReward<10>,
    environment::Return<bandit::rewards::ConstantReward<10>>>;

using NBanditPolicy = RandomPolicy<NBanditEnvironment>;

int main() {

  auto state = ExampleState{};
  auto action = ExampleAction{
      policy::random_spec_gen<typename ExampleAction::SpecType>()};
  auto nextState = ExampleStep::step(state, action);
  auto transition = ExampleTransition{
      .state = state, .action = action, .nextState = nextState};
  auto reward = ExampleReward::reward(transition);
  auto discountReward = returns::DiscountReturn<ExampleReward, 0.5F>::value(
      TransitionSequence<2, ExampleAction>{transition, transition});

  std::cout << "Hello, world! " << nextState.val << " " << reward
            << " Discount Reward " << discountReward << "\n";

  std::minstd_rand generator = {};
  // auto banditState = bandit::BanditState<10>{};
  // auto banditAction = bandit::BanditAction<10>{};
  auto banditEnv = NBanditEnvironment{generator};
  auto randomPolicy = NBanditPolicy();
  auto banditAction = randomPolicy(banditEnv.state);
  // auto greedyBanditPolicy =
  //     bandit::policy::GreedyPolicy<NBanditEnvironment, 0.1F>{};
  //
  auto trans = banditEnv.step(banditAction);
  auto reward2 = bandit::rewards::ConstantReward<10>::reward(trans);
  //
  // greedyBanditPolicy.printActionValueEstimate();
  // randomPolicy.printActionValueEstimate();
  //
  std::cout << z << "\n";
  std::cout << z1 << "\n";
  std::cout << z2 << "\n";
  std::cout << z3 << "\n";

  std::cout << banditAction << "\n";

  std::cout << reward2 << "\n";

  std::cout << action.hash() << "\n";

  return 0;
}
