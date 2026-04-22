/**
 * @file strategies_engine.cpp
 * @brief Strong anchor definitions for RTTI stability.
 * * Moving virtual destructors to a non-inline translation unit ensures
 * exactly one 'strong' copy of the type_info symbol is emitted,
 * resolving the std::bad_cast issue in nanobind/Python 3.13.
 */

#include "strategies_engine.hpp"

namespace alpha::strategies
{

    // --- TPMCR ---
    TPMCRConfig::~TPMCRConfig() = default;
    TPMCREngine::~TPMCREngine() = default;

    // --- MAERM ---
    MAERMConfig::~MAERMConfig() = default;
    MAERMEngine::~MAERMEngine() = default;

    // --- ISRC ---
    ISRCConfig::~ISRCConfig() = default;
    ISRCEngine::~ISRCEngine() = default;

    // --- VSRA ---
    VSRAConfig::~VSRAConfig() = default;
    VSRAEngine::~VSRAEngine() = default;

    // --- FDSP ---
    FDSPConfig::~FDSPConfig() = default;
    FDSPEngine::~FDSPEngine() = default;

} // namespace alpha::strategies
