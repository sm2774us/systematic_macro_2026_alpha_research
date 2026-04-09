/**
 * @file strategies_engine.hpp
 * @brief Combined signal engines for Q2-Q4 2026 alpha strategies.
 *
 * Implements four additional systematic macro strategies:
 *   - TPMCR: Term Premium Momentum & Curve Regime Signal (Rates Futures)
 *   - MAERM: Macro-Adjusted Earnings Revision Momentum (Equity Index Futures)
 *   - ISRC:  Inventory Surprise × Roll Return Composite (Energy Futures)
 *   - VSRA:  Volatility Surface Regime Arbitrage (SPX Options + VIX Futures)
 *   - FDSP:  Fiscal Dominance Shock Propagation (Cross-Asset)
 *
 * All engines follow the same interface contract:
 *   - `tick(DayData)` → Output  (incremental, online, O(N) per call)
 *   - `batch(MatrixXd)` → MatrixXd (vectorised backtest path)
 *
 * @author Alpha Research Pod — 2026
 * @copyright 2026 Alpha Research Pod. All rights reserved.
 */

#pragma once

#include "math_utils.hpp"
#include <array>
#include <cmath>
#include <expected>
#include <print>

namespace alpha::strategies {

using namespace alpha::math;

// =============================================================================
// TPMCR — Term Premium Momentum & Curve Regime Signal
// =============================================================================

/**
 * @brief Configuration for the TPMCR engine.
 */
struct TPMCRConfig {
    int    tpm_window{15};        ///< Term-premium momentum look-back (trading days)
    double crs_persistence{0.85}; ///< HMM self-transition probability (diagonal prior)
    double fiscal_threshold{1.5}; ///< Fiscal stress z-score activation threshold
    double target_vol{0.10};      ///< Annual portfolio vol target
    double max_pos{0.30};         ///< Max position per instrument
    double ridge_lambda{0.05};    ///< Ridge regularisation
    int    n_regimes{4};          ///< Number of HMM curve regimes
};

/**
 * @brief TPMCR signal engine: Term Premium Momentum × Curve Regime.
 *
 * Signal: S^TPMCR_{i,t} = β₁·TPM_{i,t} + β₂·CRS_{i,t} + β₃·FiscalStress_t·1[USD bond]
 *
 * TPM_{t}  = z_cs( TP_ACM_10Y_t - TP_ACM_10Y_{t-τ} )   [τ = 15 days]
 * CRS_{t}  = HMM state-mapped signal {-1, -0.5, +0.5, +1}
 * FSO_{t}  = z( -swap_spread_30Y + cds_5Y_proxy )
 */
class TPMCREngine {
public:
    /// Input data for one trading day
    struct DayData {
        double tp_acm_10y;       ///< ACM 10Y term premium (bps)
        double yield_2y;         ///< 2Y UST yield
        double yield_10y;        ///< 10Y UST yield
        double yield_30y;        ///< 30Y UST yield
        double swap_spread_30y;  ///< 30Y USD swap spread (fraction)
        double cds_5y_proxy;     ///< 5Y CDS proxy for fiscal stress
        std::array<double, 5> returns;  ///< [ZN, ZB, RX, G, TN] daily returns
    };

    /// Output: signals and positions for rate futures
    struct Output {
        double tpm_z;           ///< Term premium momentum z-score
        int    curve_regime;    ///< HMM regime {0:bull-flat, 1:bull-steep, 2:bear-flat, 3:bear-steep}
        double fiscal_stress_z; ///< Fiscal stress composite z-score
        std::array<double, 5> signals;   ///< Per-instrument signals
        std::array<double, 5> positions; ///< Vol-targeted positions
    };

    static constexpr int N_INSTRUMENTS = 5;  ///< ZN, ZB, RX, G, TN

    explicit TPMCREngine(TPMCRConfig cfg = {}) noexcept
        : cfg_{cfg} {}

    /**
     * @brief Process one day of rate data and compute TPMCR output.
     * @param d Day's market data.
     * @return Signal and position output.
     */
    [[nodiscard]] Output tick(const DayData& d) noexcept {
        // Track term premium in ring buffer
        tp_buf_.push(d.tp_acm_10y);

        // ── TPM: momentum over τ days ───────────────────────────────────────
        double tpm_raw = 0.0;
        if (tp_buf_.ready(cfg_.tpm_window)) {
            // Simple difference: current - τ days ago (stored in ring buffer)
            tpm_raw = d.tp_acm_10y - tp_at_lag_;
        }
        tp_lag_buf_.push(d.tp_acm_10y);
        if (tp_lag_buf_.ready(cfg_.tpm_window)) {
            tp_at_lag_ = tp_lag_buf_.back();  // oldest value in window
        }

        tpm_tracker_.update(tpm_raw);
        const double tpm_z = tpm_tracker_.ready() ? tpm_tracker_.z(tpm_raw) : 0.0;

        // ── CRS: yield curve regime classification ──────────────────────────
        // Simple rule-based approximation (production: full Baum-Welch HMM)
        const double slope_2s10s = d.yield_10y - d.yield_2y;
        const double slope_10s30s = d.yield_30y - d.yield_10y;
        const double delta_10y = d.yield_10y - prev_yield_10y_;
        prev_yield_10y_ = d.yield_10y;

        // Regime: {0:bull-flat, 1:bull-steep, 2:bear-flat, 3:bear-steep}
        int regime = 3;  // default: bear-steep (2026 base case)
        if (delta_10y < 0 && slope_2s10s > 0)       regime = 1;  // bull-steep
        else if (delta_10y < 0 && slope_2s10s <= 0)  regime = 0;  // bull-flat
        else if (delta_10y >= 0 && slope_2s10s <= 0) regime = 2;  // bear-flat
        // else bear-steep

        // CRS signal mapping per instrument [ZN, ZB, RX, G, TN]
        static constexpr std::array<std::array<double, N_INSTRUMENTS>, 4> kCRSMap = {{
            {{0.0,  1.0,  0.5,  0.5,  0.5}},   // bull-flat:  long ZB
            {{1.0,  0.5,  0.5,  0.5,  1.0}},   // bull-steep: long ZN, TN
            {{-0.5, -1.0, -0.5, -0.5, -0.5}},  // bear-flat:  short ZB
            {{-0.5, -1.0, -0.5, -0.3, -0.5}},  // bear-steep: short ZB (2026!)
        }};

        // ── Fiscal stress overlay ────────────────────────────────────────────
        const double fso_raw = -d.swap_spread_30y * 1000.0
                              + d.cds_5y_proxy * 10000.0;
        fso_tracker_.update(fso_raw);
        const double fso_z = fso_tracker_.ready() ? fso_tracker_.z(fso_raw) : 0.0;

        // ── Composite signals ────────────────────────────────────────────────
        Output out;
        out.tpm_z           = tpm_z;
        out.curve_regime    = regime;
        out.fiscal_stress_z = fso_z;

        for (int i = 0; i < N_INSTRUMENTS; ++i) {
            double sig = 0.4 * tpm_z + 0.4 * kCRSMap[regime][i];
            // Fiscal overlay only for USD bonds (ZN=0, ZB=1, TN=4)
            if (i == 0 || i == 1 || i == 4) {
                if (std::abs(fso_z) > cfg_.fiscal_threshold) {
                    sig += 0.2 * fso_z;
                }
            }
            out.signals[i] = sig;

            // Vol-targeted position
            vol_trackers_[i].update(d.returns[i]);
            const double ann_vol = std::max(
                vol_trackers_[i].vol() * std::sqrt(252.0), 0.05);
            const double pos = sig * cfg_.target_vol / ann_vol;
            out.positions[i] = std::clamp(pos, -cfg_.max_pos, cfg_.max_pos);
        }

        return out;
    }

private:
    TPMCRConfig                              cfg_;
    RingBuffer<64>                           tp_buf_;
    RingBuffer<64>                           tp_lag_buf_;
    double                                   tp_at_lag_{0.0};
    double                                   prev_yield_10y_{0.0};
    EWMATracker<30>                          tpm_tracker_{21.0};
    EWMATracker<30>                          fso_tracker_{63.0};
    std::array<EWMATracker<30>, N_INSTRUMENTS> vol_trackers_{
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}, EWMATracker<30>{21.0},
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}};
};

// =============================================================================
// MAERM — Macro-Adjusted Earnings Revision Momentum
// =============================================================================

/**
 * @brief Configuration for the MAERM engine.
 */
struct MAERMConfig {
    int    revision_window{21};    ///< EPS revision look-back (trading days)
    double ism_threshold_bull{52}; ///< ISM PMI bull-regime threshold
    double ism_threshold_bear{48}; ///< ISM PMI bear-regime threshold
    double target_vol{0.10};
    double max_pos{0.25};
    double ridge_lambda{0.05};
};

/**
 * @brief MAERM engine: Macro-Adjusted Earnings Revision Momentum.
 *
 * Signal: S^MAERM_{i,t} = α₁·ERB_{i,t} + α₂·(ISM_t × ERB_{i,t}) + α₃·PEAD_{i,t}
 * ERB  = EPS revision breadth (upgrades - downgrades) / N_analysts
 * ISM  = z-scored ISM manufacturing PMI (macro regime conditioner)
 * PEAD = Post-earnings announcement drift (24-hour window)
 */
class MAERMEngine {
public:
    static constexpr int N_INDICES = 4;  ///< ES, NQ, RTY, SX5E

    struct DayData {
        std::array<double, N_INDICES> eps_revisions;  ///< Breadth [-1, +1]
        double ism_pmi;                               ///< ISM manufacturing PMI
        std::array<double, N_INDICES> pead_signals;   ///< Post-earnings drift signals
        std::array<double, N_INDICES> returns;        ///< Daily returns
    };

    struct Output {
        std::array<double, N_INDICES> signals;
        std::array<double, N_INDICES> positions;
        double ism_regime;   ///< -1=bear, 0=neutral, +1=bull
    };

    explicit MAERMEngine(MAERMConfig cfg = {}) noexcept : cfg_{cfg} {}

    [[nodiscard]] Output tick(const DayData& d) noexcept {
        ism_tracker_.update(d.ism_pmi);
        const double ism_z = ism_tracker_.ready() ? ism_tracker_.z(d.ism_pmi) : 0.0;

        // Regime: +1=bull, 0=neutral, -1=bear
        const double regime = d.ism_pmi >= cfg_.ism_threshold_bull  ?  1.0
                            : d.ism_pmi <= cfg_.ism_threshold_bear  ? -1.0
                            :                                           0.0;

        Output out;
        out.ism_regime = regime;

        for (int i = 0; i < N_INDICES; ++i) {
            rev_trackers_[i].update(d.eps_revisions[i]);
            const double erb_z = rev_trackers_[i].ready()
                ? rev_trackers_[i].z(d.eps_revisions[i]) : 0.0;

            // Interaction term: ISM × ERB (the key non-replicable feature)
            const double interaction = ism_z * erb_z;

            double sig = 0.45 * erb_z
                       + 0.35 * interaction
                       + 0.20 * d.pead_signals[i];
            out.signals[i] = sig;

            vol_trackers_[i].update(d.returns[i]);
            const double ann_vol = std::max(
                vol_trackers_[i].vol() * std::sqrt(252.0), 0.05);
            out.positions[i] = std::clamp(
                sig * cfg_.target_vol / ann_vol, -cfg_.max_pos, cfg_.max_pos);
        }

        return out;
    }

private:
    MAERMConfig cfg_;
    EWMATracker<30> ism_tracker_{63.0};
    std::array<EWMATracker<30>, N_INDICES> rev_trackers_{
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0},
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}};
    std::array<EWMATracker<30>, N_INDICES> vol_trackers_{
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0},
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}};
};

// =============================================================================
// ISRC — Inventory Surprise × Roll Return Composite
// =============================================================================

/**
 * @brief Configuration for the ISRC engine.
 */
struct ISRCConfig {
    int    surprise_halflife{5};   ///< EIA surprise EWMA half-life (days)
    int    roll_window{21};        ///< Roll return momentum window
    double contango_threshold{0};  ///< Roll return sign threshold
    double target_vol{0.10};
    double max_pos{0.30};
};

/**
 * @brief ISRC engine: Inventory Surprise × Roll Return Composite.
 *
 * Signal: S^ISRC_{m,t} = α₁·IS_{m,t} + α₂·(IS_{m,t} × RRM_{m,t}) + α₃·OMS_{t}
 *
 * IS  = EIA weekly inventory surprise (z-scored vs 5Y seasonal avg)
 * RRM = Roll Return Momentum: recent roll return vs baseline
 * OMS = OPEC+ meeting surprise (binary event, decayed)
 *
 * Key non-linearity: backwardation × positive IS >> contango × positive IS
 */
class ISRCEngine {
public:
    static constexpr int N_ENERGY = 3;  ///< CL (crude), NG (natgas), RB (gasoline)

    struct DayData {
        std::array<double, N_ENERGY> inventory_surprises;  ///< z-scored vs seasonal
        std::array<double, N_ENERGY> roll_returns;         ///< F2-F1 return
        double opec_surprise;                               ///< Meeting surprise (0 on non-meeting)
        std::array<double, N_ENERGY> returns;              ///< Daily futures returns
    };

    struct Output {
        std::array<double, N_ENERGY> signals;
        std::array<double, N_ENERGY> positions;
        std::array<double, N_ENERGY> curve_structure;  ///< +1=backwardation, -1=contango
    };

    explicit ISRCEngine(ISRCConfig cfg = {}) noexcept : cfg_{cfg} {}

    [[nodiscard]] Output tick(const DayData& d) noexcept {
        opec_tracker_.update(d.opec_surprise);
        const double opec_decay = opec_tracker_.mean();

        Output out;

        for (int i = 0; i < N_ENERGY; ++i) {
            is_trackers_[i].update(d.inventory_surprises[i]);
            roll_trackers_[i].update(d.roll_returns[i]);

            const double is_z = is_trackers_[i].ready()
                ? is_trackers_[i].z(d.inventory_surprises[i]) : 0.0;
            const double rrm = roll_trackers_[i].mean();

            // Key interaction: IS × sign(RRM) amplification
            const double curve_sign = rrm >= cfg_.contango_threshold ? 1.0 : -1.0;
            out.curve_structure[i] = curve_sign;

            // Backwardation amplifies bullish IS; contango mutes it
            const double amplifier = curve_sign > 0 ? 1.5 : 0.5;
            const double interaction = is_z * rrm * amplifier;

            const double sig = 0.4 * is_z
                             + 0.4 * interaction
                             + 0.2 * opec_decay;
            out.signals[i] = sig;

            vol_trackers_[i].update(d.returns[i]);
            const double ann_vol = std::max(
                vol_trackers_[i].vol() * std::sqrt(252.0), 0.05);
            out.positions[i] = std::clamp(
                sig * cfg_.target_vol / ann_vol, -cfg_.max_pos, cfg_.max_pos);
        }

        return out;
    }

private:
    ISRCConfig cfg_;
    EWMATracker<10> opec_tracker_{5.0};
    std::array<EWMATracker<10>, N_ENERGY> is_trackers_{
        EWMATracker<10>{5.0}, EWMATracker<10>{5.0}, EWMATracker<10>{5.0}};
    std::array<EWMATracker<30>, N_ENERGY> roll_trackers_{
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}, EWMATracker<30>{21.0}};
    std::array<EWMATracker<30>, N_ENERGY> vol_trackers_{
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}, EWMATracker<30>{21.0}};
};

// =============================================================================
// VSRA — Volatility Surface Regime Arbitrage
// =============================================================================

/**
 * @brief Configuration for the VSRA engine.
 */
struct VSRAConfig {
    int    vrp_window{21};          ///< VRP rolling window (days)
    double tss_threshold{2.0};      ///< Term structure slope threshold for short VX
    double ska_zscore_threshold{1.5}; ///< Skew anomaly z-score threshold
    double target_vol{0.08};        ///< Lower target vol (options = higher gamma)
    double max_pos{0.20};
};

/**
 * @brief VSRA engine: Volatility Surface Regime Arbitrage.
 *
 * Three-component composite:
 *   VRP = implied_vol_atm - realized_vol  (vol risk premium, mean-reverting)
 *   TSS = VIX/VIX3M term structure slope  (carry signal for short VX)
 *   SKA = put skew - realized left-tail   (skew anomaly from tariff panic)
 *
 * Signal: S^VSRA_t = β₁·VRP_t + β₂·TSS_t + β₃·SKA_t
 */
class VSRAEngine {
public:
    struct DayData {
        double iv_atm;       ///< ATM 30-day implied vol (SPX)
        double rv_21d;       ///< 21-day realized vol
        double vix_spot;     ///< VIX spot level
        double vix_3m;       ///< 3-month VIX (VIX3M)
        double put_skew;     ///< 25-delta put skew (IV - ATM IV)
        double rv_skew;      ///< Realized left-tail skewness (22-day)
        double vx_return;    ///< VX futures daily return (for vol estimation)
    };

    struct Output {
        double vrp_z;        ///< VRP z-score
        double tss_z;        ///< Term structure slope z-score
        double ska_z;        ///< Skew anomaly z-score
        double signal;       ///< Composite signal (+ = short vol / short VX)
        double position;     ///< VX futures position (+ = short)
    };

    explicit VSRAEngine(VSRAConfig cfg = {}) noexcept : cfg_{cfg} {}

    [[nodiscard]] Output tick(const DayData& d) noexcept {
        // VRP: IV - RV (positive = implied rich → sell vol)
        const double vrp_raw = d.iv_atm - d.rv_21d;
        vrp_tracker_.update(vrp_raw);
        const double vrp_z = vrp_tracker_.ready() ? vrp_tracker_.z(vrp_raw) : 0.0;

        // TSS: VIX/VIX3M ratio (contango → short VX carry)
        const double tss_raw = d.vix_spot / std::max(d.vix_3m, 0.01);
        tss_tracker_.update(tss_raw);
        const double tss_z = tss_tracker_.ready() ? tss_tracker_.z(tss_raw) : 0.0;
        // Contango (VIX < VIX3M → ratio < 1 → tss_raw < 1 → tss_z negative)
        // Short VX when tss_z < -threshold (steep contango = rich carry)

        // SKA: put skew vs realized (positive = puts too expensive → sell puts)
        const double ska_raw = d.put_skew - d.rv_skew;
        ska_tracker_.update(ska_raw);
        const double ska_z = ska_tracker_.ready() ? ska_tracker_.z(ska_raw) : 0.0;

        // Composite (all three aligned → strong short vol signal)
        const double sig = 0.40 * vrp_z - 0.35 * tss_z + 0.25 * ska_z;
        // Note: -tss_z because steep contango (low ratio) = short VX

        // Apply activation gates
        const double gated_sig = (std::abs(vrp_z) > 0.5 ||
                                   tss_raw < (1.0 - 0.02 * cfg_.tss_threshold))
                               ? sig : 0.0;

        vx_vol_tracker_.update(d.vx_return);
        const double ann_vol = std::max(
            vx_vol_tracker_.vol() * std::sqrt(252.0), 0.05);
        const double pos = std::clamp(
            gated_sig * cfg_.target_vol / ann_vol,
            -cfg_.max_pos, cfg_.max_pos);

        return Output{
            .vrp_z    = vrp_z,
            .tss_z    = tss_z,
            .ska_z    = ska_z,
            .signal   = gated_sig,
            .position = pos,
        };
    }

private:
    VSRAConfig      cfg_;
    EWMATracker<30> vrp_tracker_{21.0};
    EWMATracker<30> tss_tracker_{21.0};
    EWMATracker<30> ska_tracker_{21.0};
    EWMATracker<30> vx_vol_tracker_{21.0};
};

// =============================================================================
// FDSP — Fiscal Dominance Shock Propagation
// =============================================================================

/**
 * @brief Configuration for the FDSP cross-asset engine.
 */
struct FDSPConfig {
    double fci_shock_threshold{1.5};   ///< FCI z-score to activate signal
    double target_vol{0.10};
    double max_pos{0.20};
    // Per-asset weights: [fci_z_weight, propagation_lag_weight]
    // Assets: TLT(bonds), UUP(USD), GLD(gold), SPY(equities), VX(vol)
    Eigen::Matrix<double, 5, 2> weights;

    FDSPConfig() {
        weights << 0.30, 0.45,
                   0.40, 0.30,
                   0.20, 0.55,
                   0.35, 0.40,
                   0.55, 0.20;
    }
};

/**
 * @brief FDSP engine: Fiscal Dominance Shock Propagation.
 *
 * Captures the cross-asset propagation sequence during debt-ceiling / fiscal
 * stress episodes via causal convolution kernels calibrated from 4 historical
 * episodes (2011, 2013, 2021, 2023).
 *
 * FCI = 0.40·(-swap_spread_30Y) + 0.35·CDS_5Y + 0.25·T-bill_spike
 *
 * Propagation kernels encode the event timeline:
 *   - VIX spikes first (day 0-2)
 *   - Gold peaks (day 5-7)
 *   - Treasuries V-shape bottom (day 7-10), then recovery
 */
class FDSPEngine {
public:
    static constexpr int N_ASSETS     = 5;   ///< TLT, UUP, GLD, SPY, VX
    static constexpr std::size_t KERNEL_LEN   = 10;  ///< Convolution kernel length

    struct DayData {
        double swap_spread_30y;              ///< 30Y swap spread (fraction)
        double cds_5y;                       ///< 5Y sovereign CDS spread
        double tbill_spike;                  ///< T-bill yield spike proxy
        std::array<double, N_ASSETS> returns; ///< Per-asset daily returns
    };

    struct Output {
        std::array<double, N_ASSETS> signals;
        std::array<double, N_ASSETS> positions;
        double fci_z;
    };

    explicit FDSPEngine(FDSPConfig cfg = {}) noexcept : cfg_{std::move(cfg)} {}

    [[nodiscard]] Output tick(const DayData& d) noexcept {
        // Build FCI composite
        const double fci_raw = 0.40 * (-d.swap_spread_30y * 1000.0)
                             + 0.35 * (d.cds_5y * 10000.0)
                             + 0.25 * (d.tbill_spike * 10000.0);
        fci_tracker_.update(fci_raw);
        const double fci_z = fci_tracker_.ready() ? fci_tracker_.z(fci_raw) : 0.0;

        fci_buf_.push(fci_z);

        Output out;
        out.fci_z = fci_z;

        for (int i = 0; i < N_ASSETS; ++i) {
            // Propagation lag via causal convolution with asset-specific kernel
            const double prop_lag = fci_buf_.ready(static_cast<int>(KERNEL_LEN / 2))
                ? fci_buf_.convolve(kPropKernels[i]) : 0.0;

            double sig = cfg_.weights(i, 0) * fci_z
                       + cfg_.weights(i, 1) * prop_lag;

            // Threshold gate: signal dampened in quiet regimes
            if (std::abs(fci_z) < cfg_.fci_shock_threshold) sig *= 0.3;

            out.signals[i] = sig;

            vol_trackers_[i].update(d.returns[i]);
            const double ann_vol = std::max(
                vol_trackers_[i].vol() * std::sqrt(252.0), 0.05);
            out.positions[i] = std::clamp(
                sig * cfg_.target_vol / ann_vol,
                -cfg_.max_pos, cfg_.max_pos);
        }

        return out;
    }

private:
    // Propagation kernels calibrated from 4 historical debt-ceiling episodes
    // Each kernel[i] = propagation profile for asset i (oldest-to-newest)
    static constexpr std::array<std::array<double, KERNEL_LEN>, N_ASSETS> kPropKernels = {{
        // TLT (bonds): V-shape — down then recovery
        {{0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3}},
        // UUP (USD): initial strength, then fades
        {{0.3, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.1, 0.0, 0.0}},
        // GLD (gold): delayed peak
        {{0.0, 0.1, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0}},
        // SPY (equities): sharp drop, slow recovery
        {{-0.2, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.2, 0.1}},
        // VX (volatility): immediate spike, fast mean-reversion
        {{0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.1, 0.0, 0.0}},
    }};

    FDSPConfig                              cfg_;
    EWMATracker<30>                         fci_tracker_{63.0};
    RingBuffer<16>                          fci_buf_;
    std::array<EWMATracker<30>, N_ASSETS>   vol_trackers_{
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}, EWMATracker<30>{21.0},
        EWMATracker<30>{21.0}, EWMATracker<30>{21.0}};
};

}  // namespace alpha::strategies
