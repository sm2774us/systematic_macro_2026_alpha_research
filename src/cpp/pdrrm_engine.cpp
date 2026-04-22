/**
 * @file pdrrm_engine.cpp
 * @brief Strong anchor definitions for RTTI stability.
 * * Moving virtual destructors to a non-inline translation unit ensures
 * exactly one 'strong' copy of the type_info symbol is emitted,
 * resolving the std::bad_cast issue in nanobind/Python 3.13.
 */

#include "pdrrm_engine.hpp"

namespace alpha::pdrrm
{

    PDRRMConfig::~PDRRMConfig() = default; // This forces a "strong" symbol in one .cpp
    PDRRMEngine::~PDRRMEngine() = default; // This forces a "strong" symbol in one .cpp

} // namespace alpha::pdrrm
