/**
 * @file pdrrm_engine.hpp
 * @brief Policy Divergence × Real Rate Momentum (PDRRM) signal engine.
 *
 * Three-component architecture:
 *   1. RRDM — Real Rate Differential Momentum (trending cross-country real rate gap)
 *   2. PSS  — Policy Surprise Score (CB meeting deviation from OIS-implied path)
 *   3. RAC  — Risk-Adjusted Carry (forward premium / realized vol)
 *
 * Combined signal: S_{i,t} = α₁·RRDM + α₂·PSS + α₃·RAC
 * Weights estimated via ridge regression on rolling 2-year in-sample window.
 *
 * Design principles:
 *   - Single-tick update path: O(N) per daily rebalance
 *   - Eigen column-major matrices for cache-friendly BLAS operations
 *   - EWMA vol trackers for online risk normalisation
 *   - std::expected return types (no exceptions on hot path)
 *
 * @author Alpha Research Pod — 2026
 * @copyright 2026 Alpha Research Pod. All rights reserved.
 */

#pragma once

#include "math_utils.hpp"
#include <array>
#include <expected>
#include <print>
#include <ranges>
#include <string_view>

namespace alpha::pdrrm {

using namespace alpha::math;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Hyper-parameters for the PDRRM signal engine.
 *
 * Default values are calibrated for G10 FX futures on CME
 * (6J, 6E, 6B, 6A, 6C, 6S, 6N) over the 2014–2025 training period.
 */
struct PDRRMConfig {
    // ── RRDM parameters ─────────────────────────────────────────────────────
    int rrdm_momentum_window{20};    ///< Momentum look-back in trading days (~4 weeks)
    int rrdm_smooth_window{5};       ///< EMA smoothing window for raw differential

    // ── PSS parameters ──────────────────────────────────────────────────────
    double pss_ewma_halflife{10.0};  ///< PSS decay half-life in trading days
    double pss_max_clip{3.0};        ///< Max z-score clip for surprise series

    // ── RAC parameters ──────────────────────────────────────────────────────
    int rac_vol_window{21};          ///< Realized vol window in trading days

    // ── Ridge regression ────────────────────────────────────────────────────
    double ridge_lambda{0.1};        ///< L2 regularisation strength

    // ── Portfolio construction ───────────────────────────────────────────────
    double target_vol{0.10};         ///< Annual portfolio volatility target (10%)
    double max_position{0.25};       ///< Max single-name position as fraction of NAV
    double tc_penalty{0.0001};       ///< Transaction cost penalty per unit turnover
    double dar_target{0.001};        ///< Dollar-at-Risk target per day (10bps)
};

// ─────────────────────────────────────────────────────────────────────────────
// PDRRM Engine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Core PDRRM signal computation engine.
 *
 * Processes panel data for N G10 FX futures instruments across T time steps.
 * Supports both batch (backtest) and incremental (live) modes.
 *
 * Thread safety: Not thread-safe. Create one instance per thread.
 */
class PDRRMEngine {
public:
    static constexpr int kMaxCurrencies = 9;  ///< G10 universe size

    /**
     * @brief Construct a PDRRM engine with given configuration.
     * @param cfg Engine hyperparameters.
     */
    explicit PDRRMEngine(PDRRMConfig cfg = {}) noexcept
        : cfg_{std::move(cfg)},
          weights_{VectorXd::Ones(3) / 3.0} {}  // Equal-weight prior

    // ── Batch signal computation ──────────────────────────────────────────────

    /**
     * @brief Compute RRDM component for a panel of currencies.
     *
     * RRDM_{i,t} = z_cs( TP_i_t - TP_i_{t - τ} )
     * where TP = nominal_rate - breakeven_inflation
     *
     * @param nominal_rates [T × N] matrix of nominal 2Y rates (annualised fraction).
     * @param breakevens    [T × N] matrix of 2Y breakeven inflation rates.
     * @return [T × N] z-scored RRDM panel, or EngineError.
     */
    [[nodiscard]] std::expected<MatrixXd, EngineError>
    compute_rrdm(const MatrixXd& nominal_rates,
                 const MatrixXd& breakevens) const noexcept {
        if (nominal_rates.rows() != breakevens.rows() ||
            nominal_rates.cols() != breakevens.cols()) [[unlikely]] {
            return std::unexpected(EngineError{
                .message = "compute_rrdm: dimension mismatch", .code = 10});
        }
        const int T = static_cast<int>(nominal_rates.rows());
        const int N = static_cast<int>(nominal_rates.cols());
        const int tau = cfg_.rrdm_momentum_window;

        // Real rates: R_real = R_nom - π_breakeven  [T × N]
        MatrixXd real_rates = nominal_rates - breakevens;

        // Differential vs USD (column 0 by convention is USD anchor)
        MatrixXd diff(T, N);
        for (int i = 0; i < N; ++i) {
            diff.col(i) = real_rates.col(i) - real_rates.col(0);
        }

        // Momentum: Δdiff over τ days
        MatrixXd rrdm(T, N);
        rrdm.topRows(tau).setZero();
        for (int t = tau; t < T; ++t) {
            rrdm.row(t) = diff.row(t) - diff.row(t - tau);
        }

        // Cross-sectional z-score each row
        for (int t = tau; t < T; ++t) {
            VectorXd row = rrdm.row(t).transpose();
            cross_sectional_zscore(row);
            rrdm.row(t) = row.transpose();
        }

        return rrdm;
    }

    /**
     * @brief Compute PSS (Policy Surprise Score) component.
     *
     * PSS_{i,t} = EWMA( CB_surprise_{i,t} ) with half-life τ_pss
     * where CB_surprise = actual_change - OIS_implied_change at meeting date.
     *
     * On non-meeting dates, surprise = 0 (no new information).
     *
     * @param cb_surprises [T × N] matrix: surprise in rate change (bp), 0 on non-meeting days.
     * @return [T × N] z-scored PSS panel, or EngineError.
     */
    [[nodiscard]] std::expected<MatrixXd, EngineError>
    compute_pss(const MatrixXd& cb_surprises) const noexcept {
        const int T = static_cast<int>(cb_surprises.rows());
        const int N = static_cast<int>(cb_surprises.cols());

        const double alpha = 1.0 - std::exp(
            -std::log(2.0) / cfg_.pss_ewma_halflife);

        MatrixXd pss(T, N);
        VectorXd ewma = VectorXd::Zero(N);

        for (int t = 0; t < T; ++t) {
            // EWMA update: only add surprise if non-zero (meeting day)
            ewma = (1.0 - alpha) * ewma +
                   alpha * cb_surprises.row(t).transpose().cwiseMin(
                       cfg_.pss_max_clip).cwiseMax(-cfg_.pss_max_clip);
            VectorXd row = ewma;
            cross_sectional_zscore(row);
            pss.row(t) = row.transpose();
        }

        return pss;
    }

    /**
     * @brief Compute RAC (Risk-Adjusted Carry) component.
     *
     * RAC_{i,t} = z_cs( ForwardPremium_{i,t} / EWMA_Vol_{i,t} )
     * where ForwardPremium = (F_{i,t} - S_{i,t}) / S_{i,t}  (covered interest parity deviation)
     *
     * @param forward_premium [T × N] matrix of annualised forward premia.
     * @param returns         [T × N] matrix of daily log-returns for vol estimation.
     * @return [T × N] z-scored RAC panel, or EngineError.
     */
    [[nodiscard]] std::expected<MatrixXd, EngineError>
    compute_rac(const MatrixXd& forward_premium,
                const MatrixXd& returns) const noexcept {
        if (forward_premium.rows() != returns.rows() ||
            forward_premium.cols() != returns.cols()) [[unlikely]] {
            return std::unexpected(EngineError{
                .message = "compute_rac: dimension mismatch", .code = 11});
        }
        const int T = static_cast<int>(returns.rows());
        const int N = static_cast<int>(returns.cols());
        const int W = cfg_.rac_vol_window;

        MatrixXd rac(T, N);

        // Rolling realized vol (annualised)
        for (int t = W; t < T; ++t) {
            VectorXd row(N);
            for (int i = 0; i < N; ++i) {
                const auto seg = returns.col(i).segment(t - W, W);
                const double rv = std::sqrt(252.0) *
                    std::sqrt(seg.array().square().mean());
                row[i] = rv > 1e-8 ? forward_premium(t, i) / rv : 0.0;
            }
            cross_sectional_zscore(row);
            rac.row(t) = row.transpose();
        }
        rac.topRows(W).setZero();

        return rac;
    }

    // ── Weight estimation ─────────────────────────────────────────────────────

    /**
     * @brief Estimate ridge regression weights from in-sample feature matrix.
     *
     * Trains on the full [T_IS × N, 3] panel (all instruments stacked).
     * Weights are stored internally for subsequent compute_positions() calls.
     *
     * @param X_is In-sample feature matrix [T_IS*N × 3] (RRDM, PSS, RAC).
     * @param y_is In-sample target vector  [T_IS*N]    (forward FX returns).
     * @return Estimated weight vector [3], or EngineError.
     */
    [[nodiscard]] std::expected<VectorXd, EngineError>
    update_weights(const MatrixXd& X_is,
                   const VectorXd& y_is) noexcept {
        auto result = ridge_regression(X_is, y_is, cfg_.ridge_lambda);
        if (result) {
            weights_ = *result;
        }
        return result;
    }

    // ── Position computation ──────────────────────────────────────────────────

    /**
     * @brief Compute optimal portfolio positions via vol-targeted Markowitz.
     *
     * Positions = vol_target / (annual_vol_i) * signal_i, then scaled
     * so portfolio annual vol = target_vol. Transaction cost penalty
     * applied as quadratic adjustment: w* = argmin(-μ'w + λ w'Σw + κ|Δw|).
     *
     * @param rrdm_now  RRDM signal vector [N] at current date.
     * @param pss_now   PSS signal vector [N] at current date.
     * @param rac_now   RAC signal vector [N] at current date.
     * @param vol_now   Per-asset annualised volatility [N].
     * @param cov       N×N annual covariance matrix.
     * @param prev_w    Previous positions [N] for TC penalty.
     * @return Optimal position vector [N], or EngineError.
     */
    [[nodiscard]] std::expected<VectorXd, EngineError>
    compute_positions(const VectorXd& rrdm_now,
                      const VectorXd& pss_now,
                      const VectorXd& rac_now,
                      const VectorXd& vol_now,
                      const MatrixXd& cov,
                      const VectorXd& prev_w) const noexcept {
        const int N = static_cast<int>(rrdm_now.size());
        if (pss_now.size() != N || rac_now.size() != N ||
            vol_now.size() != N || cov.rows() != N) [[unlikely]] {
            return std::unexpected(EngineError{
                .message = "compute_positions: dimension mismatch", .code = 20});
        }

        // Composite signal μ = α₁·RRDM + α₂·PSS + α₃·RAC
        VectorXd mu = weights_[0] * rrdm_now
                    + weights_[1] * pss_now
                    + weights_[2] * rac_now;

        // Vol-target scaling: w_i = signal_i * target_vol / vol_i
        VectorXd w_raw(N);
        for (int i = 0; i < N; ++i) {
            w_raw[i] = vol_now[i] > 1e-8
                     ? mu[i] * cfg_.dar_target / (vol_now[i] / std::sqrt(252.0))
                     : 0.0;
        }

        // Portfolio vol normalisation
        const double port_var =
            w_raw.transpose() * (cov * w_raw);
        const double port_vol = std::sqrt(std::max(port_var, 1e-16));
        if (port_vol > 1e-8) {
            w_raw *= cfg_.target_vol / port_vol;
        }

        // TC adjustment (gradient-step approximation to full QP)
        VectorXd w_adj = w_raw;
        for (int i = 0; i < N; ++i) {
            const double delta_w = w_raw[i] - prev_w[i];
            w_adj[i] -= cfg_.tc_penalty * (delta_w > 0 ? 1.0 : -1.0);
        }

        // Position clamp
        w_adj = w_adj.cwiseMax(-cfg_.max_position)
                      .cwiseMin( cfg_.max_position);

        return w_adj;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /** @return Current ridge regression weights [3]. */
    [[nodiscard]] const VectorXd& weights() const noexcept { return weights_; }

    /** @return Engine configuration. */
    [[nodiscard]] const PDRRMConfig& config() const noexcept { return cfg_; }

private:
    PDRRMConfig cfg_;       ///< Engine hyperparameters
    VectorXd    weights_;   ///< Current α = [α_RRDM, α_PSS, α_RAC]
};

}  // namespace alpha::pdrrm
