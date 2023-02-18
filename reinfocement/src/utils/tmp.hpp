#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

template <typename... input_t> using tuple_cat_t = decltype(std::tuple_cat(std::declval<input_t>()...));
