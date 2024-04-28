

/** @brief Temporal Difference methods are a combination of both Monte Carlo estimation and Dynamic Programming estimation.
 * 
 * @details TD methods combine the best elements of monte carlo (distribution free value updates) and dynamic programming
 * (bootstrapping value updates from using current estimates so as to not require awaiting the return from an episode to 
 * make an update). 
 * Both TD and Monte Carlo 'sample' future successor states. TD(0) samples a single next successor state; and not explore 
 * the whole distribution of reachable states as in Dynamic Programming.
 */
namespace temporal_difference;