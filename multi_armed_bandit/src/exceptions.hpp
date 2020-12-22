#pragma once
#include <exception>
#include <iostream>
#include <type_traits>
#include <vector>

struct UnequalVectorDimsException : public std::exception {
  const char *what() const throw() {
    return "Vectors do not share the same length";
  }
};

/** @brief Throw exception when two vectors are not of equal size. */
template <typename T1, typename T2>
void checkEqualVectorDims(std::vector<T1> *v0, std::vector<T2> *v1) {
  if (v0->size() != v1->size()) {
    throw UnequalVectorDimsException();
  }
  return;
}

/** @brief Throw exception when two vectors are not of equal size. */
template <typename T1, typename T2>
void checkEqualVectorDims(std::vector<T1> &v0, std::vector<T2> &v1) {
  if (v0.size() != v1.size()) {
    throw UnequalVectorDimsException();
  }
  return;
}

struct InvalidProbability : public std::exception {
  const char *what() const throw() {
    return "Valid probabilities are between [0,1]";
  }
};

/** @brief Throw exception when value is not a valid probability. */
template <typename T, typename = typename std::enable_if<
                          std::is_floating_point<T>::value, T>::type>
void checkValidProbability(T &a) {
  if (a < 0 or a > 1) {
    throw InvalidProbability();
  }
  return;
};