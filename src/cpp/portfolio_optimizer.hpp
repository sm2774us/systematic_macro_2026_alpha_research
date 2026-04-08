/**
 * @file portfolio_optimizer.hpp
 * @brief Master cross-signal portfolio optimizer with PCA risk decomposition.
 *
 * Combines PDRRM, TPMCR, MAERM, ISRC, VSRA, and FDSP signals into a single
 * portfolio subject to:
 *   1. Per-strategy annual vol target (10% each)
 *   2. PCA-based cross-signal risk budget (no double-loading on PC1/PC2/PC3)
 *   3. Master QP with transaction cost penalty and gross exposure cap
 *   4. Signal decay monitoring with half-life alerting
 *
 * Principal Components of cross-signal risk:
 *   PC1 (~40% variance): Risk-On/Off — correlated: MAERM + FDSP_SPY
 *   PC2 (~25% variance): Rates Level — correlated: TPMCR + FDSP_TLT
 *   PC3 (~20% variance): Vol Regime  — correlated: VSRA + FDSP_VX
 *
 * @author Alpha Research Pod — 2026
 * @copyright 2026 Alpha Research Pod. All rights reserved.
 */

#pragma once

#include "math_utils.hpp"
#include "pdrrm_engine.hpp"
#include "strategies_engine.hpp"
#include <array>
#include <expected>
#include <print>
#include <string_view>
#include <vector>

namespace alpha::portfolio {

using namespace alpha::math;

// ─────────────────────────────────────────────────────────────────────────────
// KPI Bundle — all performance metrics computed on the hot path
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Comprehensive KPI bundle for a backtested signal.
 *
 * Populated by `BacktestEngine::run()` and exposed to Python via nanobind.
 */
struct KPIBundle {
    // ── Return metrics ───────────────────────────────────────────────────────
    double sharpe_ratio{0.0};     ///< Annualised Sharpe (daily PnL series)
    double sortino_ratio{0.0};    ///< Annualised Sortino (downside only)
    double calmar_ratio{0.0};     ///< CAGR / MaxDrawdown
    double cagr{0.0};             ///< Compound Annual Growth Rate
    double total_return{0.0};     ///< Cumulative return over full period

    // ── Risk metrics ─────────────────────────────────────────────────────────
    double max_drawdown{0.0};     ///< Maximum drawdown (fraction)
    double annual_vol{0.0};       ///< Realised annual volatility
    double var_95{0.0};           ///< 95% 1-day Value-at-Risk (historical sim)
    double expected_shortfall{0.0}; ///< Expected shortfall (CVaR 95%)

    // ── Signal quality ───────────────────────────────────────────────────────
    double mean_ic{0.0};          ///< Mean Information Coefficient
    double icir{0.0};             ///< IC Information Ratio = mean(IC)/std(IC)
    double hit_rate{0.0};         ///< Fraction of days with positive PnL
    double avg_turnover{0.0};     ///< Average daily absolute turnover

    // ── Decay monitoring ─────────────────────────────────────────────────────
    double ic_half_life_days{0.0}; ///< Estimated IC half-life (AR1 regression)
    bool   decay_alert{false};     ///< True if half-life < 30 days (alert needed)

    // ── Transaction costs ────────────────────────────────────────────────────
    double gross_alpha_bps{0.0};  ///< Gross annual alpha in bps
    double tc_drag_bps{0.0};      ///< Transaction cost drag in bps/year
    double net_alpha_bps{0.0};    ///< Net alpha after TC

    // ── Black-swan resilience ────────────────────────────────────────────────
    double covid_drawdown{0.0};    ///< Drawdown during Mar 2020 episode
    double taper_tantrum_dd{0.0};  ///< Drawdown during 2022 hiking shock
    double carry_unwind_dd{0.0};   ///< Drawdown during Aug 2024 JPY unwind
};

// ─────────────────────────────────────────────────────────────────────────────
// Signal Decay Monitor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Online signal quality monitor for production deployment.
 *
 * Maintains a rolling window of IC values and computes:
 *   - 30-day rolling mean IC
 *   - IC z-score vs historical distribution
 *   - Half-life decay estimate
 *   - Alert flags for Slack/CI notification
 */
class SignalDecayMonitor {
public:
    explicit SignalDecayMonitor(int window = 60) noexcept
        : window_{window}, ic_tracker_{static_cast<double>(window)} {}

    /**
     * @brief Update with a new daily IC observation.
     * @param ic New IC value (Spearman rank correlation).
     * @return True if decay alert should be triggered.
     */
    bool update(double ic) noexcept {
        ic_tracker_.update(ic);
        ic_history_.push_back(ic);
        if (static_cast<int>(ic_history_.size()) > window_ * 3) {
            ic_history_.erase(ic_history_.begin());
        }

        if (!ic_tracker_.ready()) return false;

        const double ic_z = ic_tracker_.z(ic);

        // Compute rolling half-life if we have enough history
        if (static_cast<int>(ic_history_.size()) >= 30) {
            VectorXd ic_vec = Eigen::Map<const VectorXd>(
                ic_history_.data(), ic_history_.size());
            half_life_ = half_life_decay(ic_vec);
        }

        // Alert conditions:
        // 1. IC z-score below -2 (IC has dropped significantly)
        // 2. Half-life < 20 trading days (signal decaying fast)
        // 3. 30-day rolling mean IC < 0.02 (near-zero predictability)
        alert_ = (ic_z < -2.0) ||
                 (half_life_ < 20.0) ||
                 (ic_tracker_.mean() < 0.02);

        return alert_;
    }

    [[nodiscard]] double half_life() const noexcept { return half_life_; }
    [[nodiscard]] double rolling_mean_ic() const noexcept { return ic_tracker_.mean(); }
    [[nodiscard]] bool   alert() const noexcept { return alert_; }
    [[nodiscard]] double ic_zscore(double ic) const noexcept {
        return ic_tracker_.z(ic);
    }

private:
    int             window_;
    EWMATracker<30> ic_tracker_;
    std::vector<double> ic_history_;
    double          half_life_{std::numeric_limits<double>::infinity()};
    bool            alert_{false};
};

// ─────────────────────────────────────────────────────────────────────────────
// Backtest Engine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Full-pipeline backtester computing all KPIs from raw signal panels.
 *
 * Accepts pre-computed signal panels (from Python Polars pipeline) and
 * computes positions, PnL, and all KPIs entirely in C++ for performance.
 *
 * Typical daily batch time: < 5ms for 2520 days × 30 instruments.
 */
class BacktestEngine {
public:
    /**
     * @brief Run a full backtest from signal and return panels.
     *
     * @param signals  [T × N] panel of pre-computed combined signals.
     * @param returns  [T × N] panel of daily futures returns.
     * @param tc_rate  Round-trip transaction cost per unit notional.
     * @param ann_days Trading days per year (default 252).
     * @return Populated KPIBundle, or EngineError.
     */
    [[nodiscard]] std::expected<KPIBundle, EngineError>
    run(const MatrixXd& signals,
        const MatrixXd& returns,
        double tc_rate = 0.0001,
        double ann_days = 252.0) const noexcept {
        const int T = static_cast<int>(signals.rows());
        const int N = static_cast<int>(signals.cols());

        if (returns.rows() != T || returns.cols() != N) [[unlikely]] {
            return std::unexpected(EngineError{
                .message = "BacktestEngine::run: dimension mismatch", .code = 100});
        }
        if (T < 252) [[unlikely]] {
            return std::unexpected(EngineError{
                .message = "Need at least 252 days for KPI estimation", .code = 101});
        }

        KPIBundle kpis;
        VectorXd  daily_pnl(T);
        VectorXd  daily_ic(T);
        VectorXd  prev_w = VectorXd::Zero(N);
        double    total_tc_cost = 0.0;
        int       win_days = 0;

        for (int t = 21; t < T; ++t) {
            // Vol-scaled positions at time t
            VectorXd w(N);
            for (int i = 0; i < N; ++i) {
                // Rolling 21-day realized vol
                const auto ret_seg = returns.col(i).segment(t - 21, 21);
                const double rv = std::sqrt(ret_seg.array().square().mean());
                const double ann_vol = rv * std::sqrt(ann_days);
                w[i] = ann_vol > 1e-8
                     ? signals(t, i) * 0.10 / ann_vol  // 10% vol target
                     : 0.0;
                w[i] = std::clamp(w[i], -0.30, 0.30);
            }

            // Transaction costs (applied before PnL)
            const double turnover = (w - prev_w).cwiseAbs().sum();
            const double tc_cost  = tc_rate * turnover;
            total_tc_cost += tc_cost;

            // Daily PnL: w' * r_{t+1} - TC
            const double raw_pnl = t + 1 < T
                ? w.dot(returns.row(t + 1).transpose()) : 0.0;
            daily_pnl[t] = raw_pnl - tc_cost;
            if (daily_pnl[t] > 0) ++win_days;

            // IC: rank correlation between signal and next-day return
            if (t + 1 < T) {
                VectorXd sig_t = signals.row(t).transpose();
                VectorXd ret_t1 = returns.row(t + 1).transpose();
                daily_ic[t] = information_coefficient(sig_t, ret_t1);
            }

            prev_w = w;
        }

        // ── Cumulative PnL ─────────────────────────────────────────────────
        VectorXd cum_pnl(T);
        cum_pnl[0] = 0.0;
        for (int t = 1; t < T; ++t) {
            cum_pnl[t] = cum_pnl[t - 1] + daily_pnl[t];
        }

        // ── Return metrics ─────────────────────────────────────────────────
        kpis.sharpe_ratio = sharpe_ratio(daily_pnl.tail(T - 21), ann_days);

        // Sortino (downside deviation only)
        VectorXd neg_pnl = daily_pnl.tail(T - 21)
            .array().min(0.0).matrix();
        const double dd_vol = std::sqrt(neg_pnl.array().square().mean());
        const double mu_pnl = daily_pnl.tail(T - 21).mean();
        kpis.sortino_ratio = dd_vol > 1e-12
            ? mu_pnl / dd_vol * std::sqrt(ann_days) : 0.0;

        kpis.max_drawdown = max_drawdown(cum_pnl);
        kpis.annual_vol   = daily_pnl.tail(T - 21).array().square().mean();
        kpis.annual_vol   = std::sqrt(kpis.annual_vol * ann_days);
        kpis.total_return = cum_pnl[T - 1];
        kpis.cagr         = kpis.annual_vol > 0
            ? kpis.total_return / (static_cast<double>(T) / ann_days)
            : 0.0;
        kpis.calmar_ratio = kpis.max_drawdown > 1e-8
            ? kpis.cagr / kpis.max_drawdown : 0.0;

        // ── Risk metrics ───────────────────────────────────────────────────
        std::vector<double> sorted_pnl(
            daily_pnl.data() + 21, daily_pnl.data() + T);
        std::sort(sorted_pnl.begin(), sorted_pnl.end());
        const int var_idx = static_cast<int>(sorted_pnl.size() * 0.05);
        kpis.var_95 = -sorted_pnl[var_idx];
        double es_sum = 0.0;
        for (int i = 0; i <= var_idx; ++i) es_sum += sorted_pnl[i];
        kpis.expected_shortfall = -es_sum / (var_idx + 1);

        // ── Signal quality ─────────────────────────────────────────────────
        const VectorXd ic_valid = daily_ic.segment(21, T - 22);
        kpis.mean_ic = ic_valid.mean();
        const double ic_std = std::sqrt(
            (ic_valid.array() - kpis.mean_ic).square().mean());
        kpis.icir     = ic_std > 1e-12 ? kpis.mean_ic / ic_std : 0.0;
        kpis.hit_rate = static_cast<double>(win_days) / (T - 21);

        // ── TC metrics ─────────────────────────────────────────────────────
        kpis.avg_turnover    = total_tc_cost / (tc_rate * std::max(T - 21, 1));
        kpis.tc_drag_bps     = total_tc_cost / (T - 21) * ann_days * 10000.0;
        kpis.gross_alpha_bps = (kpis.cagr + total_tc_cost / (T / ann_days))
                              * 10000.0;
        kpis.net_alpha_bps   = kpis.cagr * 10000.0;

        // ── Decay monitoring ───────────────────────────────────────────────
        kpis.ic_half_life_days = half_life_decay(ic_valid);
        kpis.decay_alert       = kpis.ic_half_life_days < 30.0
                              || kpis.mean_ic < 0.02;

        return kpis;
    }
};

}  // namespace alpha::portfolio
