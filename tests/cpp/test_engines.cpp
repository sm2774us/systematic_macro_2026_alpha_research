/**
 * @file test_engines.cpp
 * @brief Google Test suite for all C++26 alpha signal engines.
 *
 * Test coverage:
 *   - math_utils: EWMA, RingBuffer, ridge regression, Sharpe, IC, HL decay
 *   - pdrrm_engine: RRDM, PSS, RAC, weight estimation, positions
 *   - strategies_engine: TPMCR, MAERM, ISRC, VSRA, FDSP
 *   - portfolio_optimizer: BacktestEngine, SignalDecayMonitor
 *
 * All tests use synthetic data generated with fixed seeds for reproducibility.
 *
 * @author Alpha Research Pod — 2026
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <ranges>

#include "../../src/cpp/math_utils.hpp"
#include "../../src/cpp/pdrrm_engine.hpp"
#include "../../src/cpp/strategies_engine.hpp"
#include "../../src/cpp/portfolio_optimizer.hpp"

using namespace alpha::math;
using namespace alpha::pdrrm;
using namespace alpha::strategies;
using namespace alpha::portfolio;

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Generate a synthetic T×N matrix with controlled values
MatrixXd synth_matrix(int T, int N, double scale = 0.01, uint32_t seed = 42) {
    MatrixXd m(T, N);
    uint32_t state = seed;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            state = state * 1664525u + 1013904223u;
            m(t, i) = scale * (static_cast<double>(state & 0xFFFF) / 0xFFFF - 0.5);
        }
    }
    return m;
}

/// Generate a synthetic rate matrix with mean-reverting dynamics
MatrixXd synth_rates(int T, int N, double mean = 0.03) {
    MatrixXd m(T, N);
    for (int i = 0; i < N; ++i) {
        double r = mean + 0.005 * i;
        for (int t = 0; t < T; ++t) {
            r = 0.999 * r + 0.001 * mean;
            m(t, i) = r;
        }
    }
    return m;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// math_utils tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(EWMATrackerTest, InitialState) {
    EWMATracker<30> tracker{21.0};
    EXPECT_EQ(tracker.count(), 0);
    EXPECT_FALSE(tracker.ready());
}

TEST(EWMATrackerTest, SingleUpdate) {
    EWMATracker<30> tracker{21.0};
    tracker.update(1.0);
    EXPECT_EQ(tracker.count(), 1);
    EXPECT_DOUBLE_EQ(tracker.mean(), 1.0);
}

TEST(EWMATrackerTest, ConvergesToMean) {
    EWMATracker<5> tracker{10.0};
    const double target = 5.0;
    for (int i = 0; i < 200; ++i) tracker.update(target);
    EXPECT_NEAR(tracker.mean(), target, 0.01);
}

TEST(EWMATrackerTest, ReadyAfterMinObs) {
    EWMATracker<10> tracker{21.0};
    for (int i = 0; i < 9; ++i) {
        tracker.update(1.0);
        EXPECT_FALSE(tracker.ready());
    }
    tracker.update(1.0);
    EXPECT_TRUE(tracker.ready());
}

TEST(EWMATrackerTest, ZScoreNormalized) {
    EWMATracker<10> tracker{21.0};
    for (int i = 0; i < 100; ++i) tracker.update(static_cast<double>(i % 2));
    // Z-score of an extreme value should be large
    EXPECT_GT(std::abs(tracker.z(100.0)), 1.0);
}

TEST(EWMATrackerTest, ResetClearsState) {
    EWMATracker<10> tracker{21.0};
    for (int i = 0; i < 50; ++i) tracker.update(5.0);
    EXPECT_TRUE(tracker.ready());
    tracker.reset();
    EXPECT_EQ(tracker.count(), 0);
    EXPECT_FALSE(tracker.ready());
}

TEST(RingBufferTest, PushAndBack) {
    RingBuffer<8> buf;
    buf.push(3.14);
    EXPECT_DOUBLE_EQ(buf.back(), 3.14);
    EXPECT_EQ(buf.size(), 1);
}

TEST(RingBufferTest, Wraps) {
    RingBuffer<4> buf;
    for (int i = 0; i < 10; ++i) buf.push(static_cast<double>(i));
    EXPECT_EQ(buf.back(), 9.0);
    EXPECT_EQ(buf.size(), 4);
}

TEST(RingBufferTest, ConvolveWithKernel) {
    RingBuffer<8> buf;
    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);
    std::array<double, 3> kernel = {0.5, 0.3, 0.2};
    // Convolve: newest=3.0*0.5 + 2.0*0.3 + 1.0*0.2 = 1.5+0.6+0.2=2.3
    EXPECT_NEAR(buf.convolve(kernel), 2.3, 1e-10);
}

TEST(RidgeRegressionTest, BasicFit) {
    const int T = 100, K = 3;
    MatrixXd X = MatrixXd::Random(T, K);
    VectorXd true_w(K);
    true_w << 1.0, -0.5, 0.3;
    VectorXd y = X * true_w + 0.01 * VectorXd::Random(T);

    auto result = ridge_regression(X, y, 0.01);
    ASSERT_TRUE(result.has_value());
    const VectorXd& w = *result;
    EXPECT_NEAR(w[0], 1.0, 0.05);
    EXPECT_NEAR(w[1], -0.5, 0.05);
}

TEST(RidgeRegressionTest, DimensionMismatch) {
    MatrixXd X = MatrixXd::Random(10, 3);
    VectorXd y(5);  // Wrong size
    auto result = ridge_regression(X, y);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, 1);
}

TEST(RidgeRegressionTest, NegativeLambdaError) {
    MatrixXd X = MatrixXd::Random(10, 3);
    VectorXd y(10);
    auto result = ridge_regression(X, y, -1.0);
    EXPECT_FALSE(result.has_value());
}

TEST(CrossSectionalZScoreTest, ZeroMeanUnitVar) {
    VectorXd v(5);
    v << 1.0, 2.0, 3.0, 4.0, 5.0;
    cross_sectional_zscore(v);
    EXPECT_NEAR(v.mean(), 0.0, 1e-10);
    EXPECT_NEAR(v.array().square().mean(), 1.0, 1e-6);
}

TEST(CrossSectionalZScoreTest, ConstantVectorToZero) {
    VectorXd v(4);
    v.fill(3.0);
    cross_sectional_zscore(v);
    EXPECT_NEAR(v.sum(), 0.0, 1e-10);
}

TEST(SharpeRatioTest, PositiveReturns) {
    // Constant pnl has std≈0 → Sharpe=0; use alternating positive returns so std>0
    VectorXd pnl(252);
    for (int i = 0; i < 252; ++i)
        pnl[i] = 0.001 + 0.0002 * (i % 5 - 2);  // mean=0.001, std>0
    const double sr = sharpe_ratio(pnl);
    EXPECT_GT(sr, 0.0);
}

TEST(SharpeRatioTest, ZeroReturns) {
    VectorXd pnl(252);
    pnl.setZero();
    EXPECT_DOUBLE_EQ(sharpe_ratio(pnl), 0.0);
}

TEST(MaxDrawdownTest, SingleDip) {
    VectorXd cum(10);
    cum << 0.0, 0.1, 0.2, 0.15, 0.1, 0.05, 0.1, 0.2, 0.3, 0.4;
    const double mdd = max_drawdown(cum);
    EXPECT_GT(mdd, 0.0);
    EXPECT_LT(mdd, 1.0);
}

TEST(ICTest, PerfectPositiveCorrelation) {
    VectorXd sig(10), ret(10);
    sig << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    ret << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    EXPECT_NEAR(information_coefficient(sig, ret), 1.0, 1e-6);
}

TEST(ICTest, PerfectNegativeCorrelation) {
    VectorXd sig(10), ret(10);
    sig << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    ret << 10, 9, 8, 7, 6, 5, 4, 3, 2, 1;
    EXPECT_NEAR(information_coefficient(sig, ret), -1.0, 1e-6);
}

TEST(HalfLifeTest, MeanReverting) {
    VectorXd series(100);
    // Simulate AR(1) with phi=0.95
    series[0] = 1.0;
    for (int t = 1; t < 100; ++t) {
        series[t] = 0.95 * series[t - 1];
    }
    const double hl = half_life_decay(series);
    // Expected HL = -ln(2)/ln(0.95) ≈ 13.5 days
    EXPECT_GT(hl, 5.0);
    EXPECT_LT(hl, 50.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// PDRRM engine tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(PDRRMEngineTest, ComputeRRDMShape) {
    PDRRMEngine engine;
    const int T = 100, N = 5;
    auto nom = synth_rates(T, N, 0.04);
    auto be  = synth_rates(T, N, 0.025);
    auto result = engine.compute_rrdm(nom, be);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), T);
    EXPECT_EQ(result->cols(), N);
}

TEST(PDRRMEngineTest, PSSShape) {
    PDRRMEngine engine;
    MatrixXd surprises = MatrixXd::Zero(100, 5);
    surprises(10, 1) = 0.0025;
    surprises(50, 3) = -0.0050;
    auto result = engine.compute_pss(surprises);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows(), 100);
    EXPECT_EQ(result->cols(), 5);
    // PSS should be non-zero after a meeting day
    EXPECT_NE((*result)(11, 1), 0.0);
}

TEST(PDRRMEngineTest, RACDimensionMismatch) {
    PDRRMEngine engine;
    MatrixXd fp(100, 5);
    MatrixXd ret(100, 4);  // Wrong N
    auto result = engine.compute_rac(fp, ret);
    EXPECT_FALSE(result.has_value());
}

TEST(PDRRMEngineTest, WeightEstimation) {
    PDRRMEngine engine;
    const int T = 200, N = 5;
    MatrixXd X(T * N, 3);
    X.setRandom();
    VectorXd y(T * N);
    y.setRandom();
    auto result = engine.update_weights(X, y);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 3);
    // Weights should be stored
    EXPECT_EQ(engine.weights().size(), 3);
}

TEST(PDRRMEngineTest, PositionsInBounds) {
    PDRRMConfig cfg;
    cfg.max_position = 0.25;
    PDRRMEngine engine(cfg);
    const int N = 5;
    VectorXd rrdm = VectorXd::Random(N);
    VectorXd pss  = VectorXd::Random(N);
    VectorXd rac  = VectorXd::Random(N);
    VectorXd vol  = VectorXd::Ones(N) * 0.15;
    MatrixXd cov  = MatrixXd::Identity(N, N) * 0.02;
    VectorXd prev = VectorXd::Zero(N);
    auto result = engine.compute_positions(rrdm, pss, rac, vol, cov, prev);
    ASSERT_TRUE(result.has_value());
    for (int i = 0; i < N; ++i) {
        EXPECT_LE(std::abs((*result)[i]), cfg.max_position + 1e-10);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy engine tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(TPMCREngineTest, TickOutputShape) {
    TPMCREngine engine;
    TPMCREngine::DayData d{};
    d.tp_acm_10y    = 1.2;
    d.yield_2y      = 0.048;
    d.yield_10y     = 0.045;
    d.yield_30y     = 0.047;
    d.swap_spread_30y = -0.002;
    d.cds_5y_proxy  = 0.003;
    d.returns       = {0.001, -0.002, 0.0, 0.001, -0.001};

    // Run enough ticks to warm up trackers
    TPMCREngine::Output last_out{};
    for (int t = 0; t < 50; ++t) {
        d.yield_10y += 0.0001;
        last_out = engine.tick(d);
    }
    // Regime should be one of {0, 1, 2, 3}
    EXPECT_GE(last_out.curve_regime, 0);
    EXPECT_LE(last_out.curve_regime, 3);
    // Positions should be finite
    for (int i = 0; i < TPMCREngine::N_INSTRUMENTS; ++i) {
        EXPECT_TRUE(std::isfinite(last_out.positions[i]));
    }
}

TEST(MAERMEngineTest, ISMInteractionNonLinear) {
    MAERMEngine engine;
    // Simulate bull ISM (ISM > 52)
    MAERMEngine::DayData bull_day{
        .eps_revisions = {0.1, 0.2, 0.05, 0.08},
        .ism_pmi       = 55.0,
        .pead_signals  = {0.0, 0.0, 0.0, 0.0},
        .returns       = {0.01, 0.02, 0.005, 0.008},
    };

    MAERMEngine::DayData bear_day{
        .eps_revisions = {0.1, 0.2, 0.05, 0.08},
        .ism_pmi       = 45.0,
        .pead_signals  = {0.0, 0.0, 0.0, 0.0},
        .returns       = {0.01, 0.02, 0.005, 0.008},
    };

    // Run warming period
    for (int t = 0; t < 100; ++t) {
        (void)engine.tick(t < 50 ? bull_day : bear_day);
    }
    // Bull regime should have different signals than bear
    auto bull_out = engine.tick(bull_day);
    auto bear_out = engine.tick(bear_day);
    EXPECT_NE(bull_out.ism_regime, bear_out.ism_regime);
}

TEST(ISRCEngineTest, BackwardationAmplifies) {
    ISRCEngine engine;
    ISRCEngine::Output out{};

    // Alternate low/high IS during warmup so EWMA variance stays non-zero.
    // mean≈1.0, std>0.  Last tick sends IS=3.0 → z>0.
    for (int t = 0; t < 60; ++t) {
        const double is_val = (t % 2 == 0) ? 0.2 : 1.8;  // alternating → mean≈1, std>0
        ISRCEngine::DayData d{
            .inventory_surprises = {is_val, 0.3, 0.0},
            .roll_returns        = {0.005, 0.002, -0.001},  // CL backwardation
            .opec_surprise       = 0.0,
            .returns             = {0.01, 0.005, 0.0},
        };
        out = engine.tick(d);
    }
    // Final tick: IS=3.0 >> recent mean(≈1.0) → is_z > 0 → positive signal
    ISRCEngine::DayData spike{
        .inventory_surprises = {3.0, 0.3, 0.0},
        .roll_returns        = {0.005, 0.002, -0.001},
        .opec_surprise       = 0.0,
        .returns             = {0.01, 0.005, 0.0},
    };
    out = engine.tick(spike);

    EXPECT_GT(out.signals[0], 0.0);
    EXPECT_EQ(out.curve_structure[0], 1.0);  // backwardation (positive roll)
}

TEST(VSRAEngineTest, HighVRPPositiveSignal) {
    VSRAEngine engine;
    VSRAEngine::Output out{};

    // Warmup with LOW VRP (iv≈rv) so tracker mean≈0.02, std>0 from variation
    for (int t = 0; t < 60; ++t) {
        const double iv = (t % 2 == 0) ? 0.10 : 0.14;  // alternate → mean≈0.12, std>0
        out = engine.tick({
            .iv_atm    = iv,
            .rv_21d    = 0.10,
            .vix_spot  = 12.0,
            .vix_3m    = 14.0,
            .put_skew  = 0.05,
            .rv_skew   = -0.20,
            .vx_return = 0.0,
        });
    }
    // Spike: HIGH VRP (iv=0.35, rv=0.10 → vrp=0.25 >> recent mean≈0.02-0.04) → vrp_z >> 0
    out = engine.tick({
        .iv_atm    = 0.35,
        .rv_21d    = 0.10,
        .vix_spot  = 35.0,
        .vix_3m    = 37.0,
        .put_skew  = 0.07,
        .rv_skew   = -0.20,
        .vx_return = 0.0,
    });

    EXPECT_GT(out.vrp_z, 0.0);
    EXPECT_TRUE(std::isfinite(out.signal));
}

TEST(FDSPEngineTest, FiscalStressActivates) {
    FDSPEngine engine;
    // Baseline (quiet)
    FDSPEngine::DayData quiet{
        .swap_spread_30y = -0.001,
        .cds_5y          = 0.002,
        .tbill_spike     = 0.0,
        .returns         = {0.0, 0.0, 0.0, 0.0, 0.0},
    };
    // Stress spike
    FDSPEngine::DayData stress{
        .swap_spread_30y = -0.005,
        .cds_5y          = 0.020,
        .tbill_spike     = 0.010,
        .returns         = {0.0, 0.0, 0.002, -0.03, 0.0},
    };
    FDSPEngine::Output out_q{}, out_s{};
    for (int t = 0; t < 100; ++t) out_q = engine.tick(quiet);
    for (int t = 0; t < 5; ++t)   out_s = engine.tick(stress);

    EXPECT_GT(std::abs(out_s.fci_z), std::abs(out_q.fci_z));
}

// ─────────────────────────────────────────────────────────────────────────────
// BacktestEngine tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(BacktestEngineTest, BasicRun) {
    BacktestEngine bt;
    const int T = 500, N = 5;
    MatrixXd signals = synth_matrix(T, N, 1.0);
    MatrixXd returns = synth_matrix(T, N, 0.01);

    auto result = bt.run(signals, returns, 0.0001, 252.0);
    ASSERT_TRUE(result.has_value());
    const KPIBundle& kpis = *result;

    EXPECT_TRUE(std::isfinite(kpis.sharpe_ratio));
    // max_drawdown now always in [0, 1) by construction (dd/(|peak|+dd+eps))
    EXPECT_GE(kpis.max_drawdown, 0.0);
    EXPECT_LE(kpis.max_drawdown, 1.0);
    EXPECT_GE(kpis.hit_rate, 0.0);
    EXPECT_LE(kpis.hit_rate, 1.0);
}

TEST(BacktestEngineTest, DimensionMismatch) {
    BacktestEngine bt;
    MatrixXd signals(500, 5);
    MatrixXd returns(500, 4);  // Different N
    auto result = bt.run(signals, returns);
    EXPECT_FALSE(result.has_value());
}

TEST(BacktestEngineTest, TooFewDays) {
    BacktestEngine bt;
    MatrixXd s(50, 3);
    MatrixXd r(50, 3);
    auto result = bt.run(s, r);
    EXPECT_FALSE(result.has_value());
}

TEST(BacktestEngineTest, PositiveAlphaSignalShowsPositiveSharpe) {
    BacktestEngine bt;
    const int T = 600, N = 5;

    // Construct a signal that perfectly predicts next-day returns (idealized)
    MatrixXd returns = synth_matrix(T, N, 0.005);
    MatrixXd signals(T, N);
    // Lag signals by 1 day: signal_t = return_{t+1}
    signals.topRows(T - 1) = returns.bottomRows(T - 1);
    signals.row(T - 1) = returns.row(T - 1);

    auto result = bt.run(signals, returns, 0.0, 252.0);
    ASSERT_TRUE(result.has_value());
    // A predictive signal should generate positive IC
    EXPECT_GT(result->mean_ic, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// SignalDecayMonitor tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(DecayMonitorTest, NoAlertOnGoodIC) {
    SignalDecayMonitor monitor{60};
    for (int i = 0; i < 60; ++i) {
        bool alert = monitor.update(0.06);  // IC = 6%, healthy
        if (i == 59) EXPECT_FALSE(alert);
    }
    EXPECT_GT(monitor.rolling_mean_ic(), 0.02);
}

TEST(DecayMonitorTest, AlertOnDecayingIC) {
    // window=10 → EWMA half-life=10 days (alpha≈0.067, fast decay)
    // After 60 good IC then 80 bad IC: mean ≈ 0.005 << 0.02 threshold → alert
    SignalDecayMonitor monitor{10};
    for (int i = 0; i < 60; ++i) monitor.update(0.06);
    bool alert = false;
    for (int i = 0; i < 80; ++i) {
        alert = monitor.update(0.005);
    }
    EXPECT_TRUE(alert);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
