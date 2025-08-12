#pragma once

#include "containers.h"
#include <fstream>
#ifdef HAVE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#else
#include "yirage/compat/nlohmann/json.hpp"
#endif

#ifdef __CUDACC__
#include <vector_types.h>
#else
#include "yirage/compat/vector_types.h"
#endif

using json = nlohmann::json;

void to_json(json &j, int3 const &i);
void from_json(json const &j, int3 &i);
void to_json(json &j, dim3 const &i);
void from_json(json const &j, dim3 &i);

template <typename T>
T load_json(char const *file_path) {
  std::ifstream ifs(file_path);
  json j;
  ifs >> j;
  return j.get<T>();
}