/**
 * @file portfolio_optimizer.cpp
 * @brief Strong anchor definitions for RTTI stability.
 * * Moving virtual destructors to a non-inline translation unit ensures
 * exactly one 'strong' copy of the type_info symbol is emitted,
 * resolving the std::bad_cast issue in nanobind/Python 3.13.
 */

#include "portfolio_optimizer.hpp"

namespace alpha::portfolio
{
    KPIBundle::~KPIBundle() = default; // This forces a "strong" symbol in one .cpp

    SignalDecayMonitor::~SignalDecayMonitor() = default; // This forces a "strong" symbol in one .cpp

    BacktestEngine::~BacktestEngine() = default; // This forces a "strong" symbol in one .cpp
} // namespace alpha::portfolio
