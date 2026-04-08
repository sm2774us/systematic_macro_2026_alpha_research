/**
 * @file math_utils.hpp
 * @brief Low-latency, SIMD-aware mathematical primitives for alpha signal engines.
 *
 * All containers follow data-oriented design principles:
 *   - Cache-line aligned storage (64-byte alignment)
 *   - Column-major Eigen matrices for BLAS compatibility
 *   - EWMA trackers as single-pass recurrences (O(1) per tick)
 *   - Lock-free ring buffers for sliding-window operations
 *
 * Compiler hints:
 *   - [[likely]] / [[unlikely]] on hot paths
 *   - [[nodiscard]] on all value-returning functions
 *   - std::assume to aid vectorisation
 *
 * @author Alpha Research Pod — 2026
 * @copyright 2026 Alpha Research Pod. All rights reserved.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <expected>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Eigen — column-major, SIMD-enabled linear algebra
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace alpha::math {

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────

using MatrixXd = Eigen::MatrixXd;   ///< Dynamic double matrix (column-major)
using VectorXd = Eigen::VectorXd;   ///< Dynamic double vector
using Matrix3d = Eigen::Matrix3d;   ///< 3×3 double matrix

// ─────────────────────────────────────────────────────────────────────────────
// Error type for std::expected return values
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Lightweight error descriptor for alpha engine functions.
 *
 * Avoids exceptions on hot paths. Functions return
 * `std::expected<T, EngineError>` instead of throwing.
 */
struct EngineError {
    std::string message;    ///< Human-readable error description
    int         code{-1};  ///< Optional numeric error code
};

// ─────────────────────────────────────────────────────────────────────────────
// EWMA Tracker — O(1) per update, cache-hot single scalar state
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Exponentially-weighted moving average tracker with variance estimation.
 *
 * Maintains mean and variance in a single-pass O(1) recurrence using
 * Welford's online algorithm modified for exponential weighting.
 *
 * Memory layout: 4 doubles (32 bytes) → fits in a single cache line.
 *
 * @tparam kMinObs Minimum observations before `ready()` returns true.
 */
template<int kMinObs = 30>
class EWMATracker {
public:
    /**
     * @brief Construct an EWMA tracker with given half-life.
     * @param half_life_days Half-life in days; alpha = 1 - exp(-ln2/half_life).
     */
    explicit EWMATracker(double half_life_days) noexcept
        : alpha_{1.0 - std::exp(-std::log(2.0) / half_life_days)} {}

    /**
     * @brief Update tracker with a new observation.
     * @param x New data point.
     */
    void update(double x) noexcept {
        ++n_;
        if (n_ == 1) [[unlikely]] {
            mean_ = x;
            var_  = 0.0;
            return;
        }
        const double delta = x - mean_;
        mean_ += alpha_ * delta;
        var_   = (1.0 - alpha_) * (var_ + alpha_ * delta * delta);
    }

    /** @return Exponentially-weighted mean. */
    [[nodiscard]] double mean() const noexcept { return mean_; }

    /** @return Exponentially-weighted standard deviation. */
    [[nodiscard]] double vol() const noexcept { return std::sqrt(var_); }

    /** @return Standardised z-score of value x. */
    [[nodiscard]] double z(double x) const noexcept {
        const double s = std::sqrt(var_);
        return s > 1e-12 ? (x - mean_) / s : 0.0;
    }

    /** @return True if enough observations for reliable statistics. */
    [[nodiscard]] bool ready() const noexcept { return n_ >= kMinObs; }

    /** @return Number of observations seen so far. */
    [[nodiscard]] int count() const noexcept { return n_; }

    /** @brief Reset tracker to initial state. */
    void reset() noexcept { n_ = 0; mean_ = 0.0; var_ = 0.0; }

private:
    double alpha_;          ///< EWMA decay factor
    double mean_{0.0};      ///< Current EWMA mean
    double var_{0.0};       ///< Current EWMA variance
    int    n_{0};           ///< Observation count
};

// ─────────────────────────────────────────────────────────────────────────────
// Ring Buffer — lock-free sliding window for convolution / rolling stats
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Fixed-size lock-free ring buffer for streaming computations.
 *
 * Uses power-of-2 capacity for branchless modular indexing.
 * Cache-line padded to avoid false sharing in multi-threaded contexts.
 *
 * @tparam N Capacity (should be power of 2 for fast modular arithmetic).
 */
template<int N>
class RingBuffer {
    static_assert(N > 0 && (N & (N - 1)) == 0,
                  "RingBuffer capacity must be a power of 2");
public:
    RingBuffer() noexcept : head_{0}, count_{0} { data_.fill(0.0); }

    /**
     * @brief Push a new value, overwriting the oldest if full.
     * @param x Value to push.
     */
    void push(double x) noexcept {
        data_[head_ & (N - 1)] = x;
        ++head_;
        if (count_ < N) ++count_;
    }

    /**
     * @brief Compute dot-product of the buffer with a fixed kernel.
     * @tparam K Kernel length.
     * @param kernel Kernel coefficients (oldest-to-newest order).
     * @return Convolution result.
     */
    template<int K>
    [[nodiscard]] double convolve(const std::array<double, K>& kernel) const noexcept {
        double acc = 0.0;
        for (int i = 0; i < K && i < count_; ++i) {
            const int idx = (head_ - 1 - i) & (N - 1);
            acc += kernel[i] * data_[idx];
        }
        return acc;
    }

    /** @return Most recent value. */
    [[nodiscard]] double back() const noexcept {
        return data_[(head_ - 1) & (N - 1)];
    }

    /** @return Number of elements currently stored. */
    [[nodiscard]] int size() const noexcept { return count_; }

    /** @return True if buffer has at least `min_elems` elements. */
    [[nodiscard]] bool ready(int min_elems) const noexcept {
        return count_ >= min_elems;
    }

private:
    alignas(64) std::array<double, N> data_;  ///< Cache-aligned storage
    int head_{0};    ///< Write pointer (wraps via & (N-1))
    int count_{0};   ///< Current fill level
};

// ─────────────────────────────────────────────────────────────────────────────
// Ridge Regression — O(T·K² + K³) full solve
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute ridge regression weights: β = (X'X + λI)⁻¹ X'y.
 *
 * Uses Eigen's LDLT decomposition for numerical stability.
 * Input matrices must be column-major (Eigen default).
 *
 * @param X Feature matrix [T × K] (T observations, K features).
 * @param y Target vector [T].
 * @param lambda Ridge regularisation parameter (λ > 0).
 * @return K-dimensional weight vector, or EngineError on failure.
 */
[[nodiscard]] inline std::expected<VectorXd, EngineError>
ridge_regression(const MatrixXd& X,
                 const VectorXd& y,
                 double lambda = 0.1) noexcept {
    if (X.rows() != y.size()) [[unlikely]] {
        return std::unexpected(EngineError{
            .message = "X.rows() != y.size()", .code = 1});
    }
    if (lambda < 0.0) [[unlikely]] {
        return std::unexpected(EngineError{
            .message = "Ridge lambda must be non-negative", .code = 2});
    }
    const int K = static_cast<int>(X.cols());
    const MatrixXd XtX = X.transpose() * X;
    const MatrixXd reg = XtX + lambda * MatrixXd::Identity(K, K);
    const VectorXd Xty = X.transpose() * y;

    Eigen::LDLT<MatrixXd> ldlt(reg);
    if (ldlt.info() != Eigen::Success) [[unlikely]] {
        return std::unexpected(EngineError{
            .message = "LDLT decomposition failed", .code = 3});
    }
    return ldlt.solve(Xty);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-sectional z-score
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Standardise a vector to zero mean and unit variance (cross-sectional).
 *
 * @param v Input vector (modified in-place).
 * @param epsilon Minimum standard deviation to avoid division by zero.
 */
inline void cross_sectional_zscore(VectorXd& v,
                                    double epsilon = 1e-8) noexcept {
    const double mu  = v.mean();
    const double sig = std::sqrt((v.array() - mu).square().mean());
    if (sig > epsilon) {
        v = (v.array() - mu) / sig;
    } else {
        v.setZero();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Annualised Sharpe ratio
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute annualised Sharpe ratio from a daily P&L series.
 *
 * @param pnl Daily P&L vector.
 * @param ann_factor Annualisation factor (default 252 for trading days).
 * @return Sharpe ratio, or 0 if not enough data.
 */
[[nodiscard]] inline double sharpe_ratio(const VectorXd& pnl,
                                          double ann_factor = 252.0) noexcept {
    if (pnl.size() < 2) return 0.0;
    const double mu  = pnl.mean();
    const double sig = std::sqrt((pnl.array() - mu).square().mean());
    return sig > 1e-12 ? mu / sig * std::sqrt(ann_factor) : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Maximum drawdown
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute maximum drawdown from a cumulative P&L series.
 *
 * @param cum_pnl Cumulative P&L vector (not daily returns).
 * @return Maximum drawdown as a positive fraction (e.g., 0.15 = 15% MDD).
 */
[[nodiscard]] inline double max_drawdown(const VectorXd& cum_pnl) noexcept {
    double peak = cum_pnl[0];
    double mdd  = 0.0;
    for (int t = 1; t < static_cast<int>(cum_pnl.size()); ++t) {
        peak = std::max(peak, cum_pnl[t]);
        mdd  = std::max(mdd, (peak - cum_pnl[t]) / (std::abs(peak) + 1e-12));
    }
    return mdd;
}

// ─────────────────────────────────────────────────────────────────────────────
// Information Coefficient (IC)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute Spearman rank correlation (Information Coefficient).
 *
 * @param signals Signal vector [N].
 * @param returns Subsequent returns vector [N].
 * @return Spearman IC ∈ [-1, +1].
 */
[[nodiscard]] inline double information_coefficient(
    const VectorXd& signals,
    const VectorXd& returns) noexcept {
    const int N = static_cast<int>(signals.size());
    if (N < 2) return 0.0;

    // Rank both vectors
    auto rank_vec = [&](const VectorXd& v) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return v[a] < v[b]; });
        VectorXd ranks(N);
        for (int i = 0; i < N; ++i) ranks[idx[i]] = static_cast<double>(i);
        return ranks;
    };

    const VectorXd rs = rank_vec(signals);
    const VectorXd rr = rank_vec(returns);

    const double mu_s = rs.mean(), mu_r = rr.mean();
    const double cov  = ((rs.array() - mu_s) * (rr.array() - mu_r)).mean();
    const double ss   = std::sqrt((rs.array() - mu_s).square().mean());
    const double sr   = std::sqrt((rr.array() - mu_r).square().mean());
    return (ss > 1e-12 && sr > 1e-12) ? cov / (ss * sr) : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Half-life decay estimation (AR(1) regression)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Estimate the half-life of a mean-reverting signal.
 *
 * Regresses Δy_t = θ·y_{t-1} + ε and computes HL = -ln(2)/θ.
 * Used for signal decay monitoring in CI pipeline.
 *
 * @param series Time series of signal strength / IC values.
 * @return Half-life in same units as the input frequency.
 */
[[nodiscard]] inline double half_life_decay(const VectorXd& series) noexcept {
    const int T = static_cast<int>(series.size());
    if (T < 10) return std::numeric_limits<double>::infinity();

    VectorXd dy = series.tail(T - 1) - series.head(T - 1);
    VectorXd y_lag = series.head(T - 1);

    // OLS: dy = θ * y_lag
    const double theta = y_lag.dot(dy) / (y_lag.dot(y_lag) + 1e-12);
    if (theta >= 0.0) return std::numeric_limits<double>::infinity();
    return -std::log(2.0) / theta;
}

}  // namespace alpha::math
