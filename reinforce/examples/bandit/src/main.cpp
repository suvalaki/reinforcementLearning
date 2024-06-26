#include <ctime>
#include <random>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <reinforce/action.hpp>
#include <reinforce/policy/epsilon_greedy_policy.hpp>
#include <reinforce/spec.hpp>
// #include "policy/greedy_policy.hpp"
#include <reinforce/policy/finite/greedy_policy.hpp>
#include <reinforce/policy/finite/random_policy.hpp>
#include <reinforce/policy/finite/value_policy.hpp>
#include <reinforce/policy/objectives/finite_value.hpp>
#include <reinforce/policy/objectives/value_function_keymaker.hpp>
#include <reinforce/policy/policy.hpp>
#include <reinforce/policy/random_policy.hpp>

#include "bandit.hpp"
#include "bandit_environment.hpp"
#include "bandit_policy.hpp"

using namespace environment;
using namespace policy;

// Examples
struct ExampleState : State<float> {
  PrecisionType val;
};

using ExampleSpec0 = spec::BoundedAarraySpec<float, -1.0F, 1.0F, 10, 20>;
using EnumSpec0 = spec::CategoricalArraySpec<bandit::BanditActionChoices, 4, 10>;

auto &engine = xt::random::get_default_random_engine();

auto s = 123;
auto z = ::policy::random_spec_gen<ExampleSpec0>();
auto z1 = ::policy::random_spec_gen<EnumSpec0>();
auto z2 = ::policy::random_spec_gen<EnumSpec0>();
auto z3 = ::policy::random_spec_gen<EnumSpec0>();

static_assert(spec::isBoundedArraySpec<ExampleSpec0>);

using ExampleCombinedSpec = spec::CompositeArraySpec<EnumSpec0>;
using ExampleCombinedSpec1 = spec::CompositeArraySpec<ExampleSpec0>;

// using banditAction = bandit::BanditActionSpec<10>;

struct ExampleAction : Action<ExampleState, ExampleCombinedSpec> {
  using Action<ExampleState, ExampleCombinedSpec>::Action;
};

using ExampleStep = Step<ExampleAction>;
using ExampleTransition = Transition<ExampleAction>;
using ExampleSequence = TransitionSequence<2, ExampleAction>;
struct ExampleReward : Reward<ExampleAction> {
  static PrecisionType reward(const TransitionType &t) { return 10.0F; }
};
using ExampleReturn = Return<ExampleReward>;

using constatntR = bandit::rewards::ConstantReward<2>;
using constatntRet = environment::Return<bandit::rewards::ConstantReward<2>>;
using NBanditEnvironment = bandit::
    BanditEnvironment<3, bandit::rewards::ConstantReward<3>, environment::Return<bandit::rewards::ConstantReward<3>>>;

//  Tesing objects
using BanditRandomFinitePolicy = policy::FiniteRandomPolicy<NBanditEnvironment>;
using BanditKeyMaker = policy::objectives::ActionKeymaker<NBanditEnvironment>;
using BanditValue = policy::objectives::FiniteValue<NBanditEnvironment>;
using BanditValueFunction = policy::objectives::ValueFunction<BanditKeyMaker, BanditValue>;
// using BanditFiniteValueFunction = policy::objectives::FiniteValueFunction<BanditValueFunction>;
using BanditFiniteValueFunction = policy::objectives::FiniteValueFunctionHelper<BanditKeyMaker, BanditValue>;
// using BanditFinitePolicyValueFunctionMixin = policy::FinitePolicyValueFunctionMixin<BanditKeyMaker, BanditValue>;
using BanditGreedy = policy::FiniteGreedyPolicy<BanditFiniteValueFunction>;
using BanditGreedyC = policy::FiniteGreedyPolicyC<NBanditEnvironment, policy::objectives::ActionKeymaker>;

// using NBanditPolicy = RandomPolicy<NBanditEnvironment>;
// using KeyHashMappings = bandit::BanditStateActionKeymapper<NBanditEnvironment>;
// // using BanditGreedy = policy::FiniteGreedyPolicy<NBanditEnvironment,
// // KeyHashMappings>;
// using BanditGreedy = bandit::GreedyBanditPolicy<NBanditEnvironment>;
// using CIBanditGreedy = bandit::UpperConfidenceBoundGreedyBanditPolicy<NBanditEnvironment, 0.5F>;

int main() {

  std::minstd_rand generator = {};
  auto banditEnv = NBanditEnvironment{generator};
  banditEnv.printDistributions();

  std::cout << "nStates: " << banditEnv.nStates << " "
            << "nActions: " << banditEnv.nActions << "\n";
  auto randomPolicy = BanditRandomFinitePolicy();
  // auto map = typename BanditGreedy::ValueTableType();
  // std::cout << map.size() << "\n";

  auto banditValue = BanditValue{};
  auto banditFiniteValueFunction = BanditFiniteValueFunction{};

  banditFiniteValueFunction.initialize(banditEnv);

  std::cout << banditFiniteValueFunction.size() << "\n";
  banditFiniteValueFunction.prettyPrint();

  auto banditGreedy = BanditGreedyC{};

  auto action = randomPolicy(banditEnv, banditEnv.state);
  std::cout << action << "\n";

  auto key = BanditKeyMaker::make(banditEnv, banditEnv.state, action);
  std::cout << key << "\n";

  std::cout << "RANDOM ACTIONS\n";
  for (int i = 0; i < 100; i++) {
    auto recommendedAction = randomPolicy(banditEnv, banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    banditGreedy.update(banditEnv, transition);
    banditEnv.update(transition);
  }

  banditGreedy.prettyPrint();

  // auto m = keymaker.make(env, {}, {});

  /*
  auto state = ExampleState{};
  auto action = ExampleAction{policy::random_spec_gen<typename ExampleAction::SpecType>()};
  auto nextState = ExampleStep::step(state, action);
  auto transition = ExampleTransition{.state = state, .action = action, .nextState = nextState};
  auto reward = ExampleReward::reward(transition);
  auto discountReward =
      returns::DiscountReturn<ExampleReward, 0.5F>::value(TransitionSequence<2, ExampleAction>{transition, transition});

  std::cout << "Hello, world! " << nextState.val << " " << reward << " Discount Reward " << discountReward << "\n";

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
  auto reward2 = bandit::rewards::ConstantReward<3>::reward(trans);
  //
  // randomPolicy.printActionValueEstimate();
  //
  std::cout << z << "\n";
  std::cout << z1 << "\n";
  std::cout << z2 << "\n";
  std::cout << z3 << "\n";

  std::cout << banditAction << "\n";

  std::cout << reward2 << "\n";

  std::cout << action.hash() << "\n";
  std::cout << state.hash() << "\n";

  auto temp = state == state;
  auto temp1 = action == action;

  auto tmpAction = policy::random_spec_gen<ExampleCombinedSpec1>();

  auto val = tmpAction == tmpAction;

  auto banditGreedy = BanditGreedy();
  auto banditRandom = RandomPolicy<NBanditEnvironment>();

  // Prepoluate the action value estimates with some random actions
  banditEnv.reset();
  std::cout << banditEnv.state << "\n";

  auto recommendedAction = banditRandom(banditEnv.state);
  auto newState = recommendedAction.step(banditEnv.state);

  std::cout << newState << "\n";

  std::cout << "RANDOM ACTIONS\n";
  for (int i = 0; i < 100; i++) {
    auto recommendedAction = banditRandom(banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    banditGreedy.update(transition); // seed the greedy action
                                     //
    // std::cout << banditEnv.state << " " << transition.action << " "
    //           << transition.nextState << "\n";

    // std::cout << transition.state << transition.action << " "
    //           << recommendedAction << " " << banditGreedy.greedyValue() <<
    //           "\n";

    banditEnv.update(transition);
  }

  banditGreedy.printQTable();

  // Start the greedy loop
  std::cout << "GREEDY ACTIONS\n";
  for (int i = 0; i < 1000; i++) {
    auto recommendedAction = banditGreedy(banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    banditGreedy.update(transition);
    banditEnv.update(transition);
    // std::cout << transition.state << transition.action << " "
    //           << recommendedAction << " " << banditGreedy.greedyValue() <<
    //           "\n";
  }

  banditGreedy.printQTable();

  // Create an epsilon  greedy policy and run the bandit over them.
  auto epsilonGreedy = policy::EpsilonGreedyPolicy<NBanditEnvironment, BanditGreedy>{0.1F};
  std::cout << "EPSILON GREEDY ACTIONS\n";
  for (int i = 0; i < 100000; i++) {
    auto recommendedAction = epsilonGreedy(banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    epsilonGreedy.update(transition);
    banditEnv.update(transition);
    // std::cout << transition.state << transition.action << " "
    //           << recommendedAction << " " << banditGreedy.greedyValue() <<
    //           "\n";
  }

  epsilonGreedy.printQTable();
  banditEnv.printDistributions();

  // Create a UCB policy and run the bandit over them.
  auto ucbGreedy = CIBanditGreedy{};
  std::cout << "UCB GREEDY ACTIONS\n";
  for (int i = 0; i < 100000; i++) {
    auto recommendedAction = ucbGreedy(banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    ucbGreedy.update(transition);
    banditEnv.update(transition);
    // std::cout << transition.state << transition.action << " "
    //           << recommendedAction << " " << banditGreedy.greedyValue() <<
    //           "\n";
  }

  ucbGreedy.printQTable();
  banditEnv.printDistributions();

  // Create a UBC epsilon greedy policy and run the bandit over them.
  auto ucbEpsilonGreedy = policy::EpsilonGreedyPolicy<NBanditEnvironment, CIBanditGreedy>{0.1F};
  std::cout << "UCB EPSILON GREEDY ACTIONS\n";
  for (int i = 0; i < 100000; i++) {
    auto recommendedAction = ucbEpsilonGreedy(banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    ucbEpsilonGreedy.update(transition);
    banditEnv.update(transition);
    // std::cout << transition.state << transition.action << " "
    //           << recommendedAction << " " << banditGreedy.greedyValue() <<
    //           "\n";
  }

  ucbEpsilonGreedy.printQTable();
  banditEnv.printDistributions();

  // Create a distribution policy and run the bandit over them.
  auto distributionPolicy = BanditDistributionPolicy<NBanditEnvironment>{};
  std::cout << "DISTRIBUTION ACTIONS\n";
  std::cout << "Initialising with random actions \n";
  for (int i = 0; i < 100; i++) {
    auto recommendedAction = banditRandom(banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    distributionPolicy.update(banditEnv, transition);
    banditEnv.update(transition);
  }
  for (int i = 0; i < 10000; i++) {
    auto recommendedAction = distributionPolicy(banditEnv, banditEnv.state);
    auto transition = banditEnv.step(recommendedAction);
    distributionPolicy.update(banditEnv, transition);
    banditEnv.update(transition);
    // std::cout << transition.state << transition.action << " "
    //           << recommendedAction << " " << banditGreedy.greedyValue() <<
    //           "\n";
  }

  distributionPolicy.printQTable(banditEnv);
  banditEnv.printDistributions();

  */
  return 0;
}
