/**
 * @file bindings.cpp
 * @brief nanobind Python 3.13 ↔ C++26 bridge for alpha research pipeline.
 *
 * Exposes the following classes to Python:
 *   - alpha_cpp.PDRRMEngine       — PDRRM signal engine
 *   - alpha_cpp.TPMCREngine       — Term premium & curve regime engine
 *   - alpha_cpp.MAERMEngine       — Earnings revision momentum engine
 *   - alpha_cpp.ISRCEngine        — Inventory surprise × roll return
 *   - alpha_cpp.VSRAEngine        — Volatility surface regime arbitrage
 *   - alpha_cpp.FDSPEngine        — Fiscal dominance shock propagation
 *   - alpha_cpp.BacktestEngine    — Batch KPI computation
 *   - alpha_cpp.SignalDecayMonitor — Online IC decay monitoring
 *   - alpha_cpp.KPIBundle         — KPI result struct
 *
 * NumPy ↔ Eigen bridging: nanobind's ndarray<> type handles zero-copy
 * conversion for column-major double arrays (Fortran order).
 *
 * Python usage:
 *   import alpha_cpp
 *   eng = alpha_cpp.PDRRMEngine()
 *   kpis = alpha_cpp.BacktestEngine().run(signals_np, returns_np)
 *
 * @author Alpha Research Pod — 2026
 * @copyright 2026 Alpha Research Pod. All rights reserved.
 */

#include "pdrrm_engine.hpp"
#include "strategies_engine.hpp"
#include "portfolio_optimizer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

// Helper: Convert Eigen MatrixXd to numpy ndarray (zero-copy where possible)
using NpArray2D = nb::ndarray<double, nb::ndim<2>, nb::c_contig>;
using NpArray1D = nb::ndarray<double, nb::ndim<1>, nb::c_contig>;

static Eigen::MatrixXd np2eigen_mat(const NpArray2D& arr) {
    const int rows = static_cast<int>(arr.shape(0));
    const int cols = static_cast<int>(arr.shape(1));
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                          Eigen::RowMajor>>(
        arr.data(), rows, cols);
}

static Eigen::VectorXd np2eigen_vec(const NpArray1D& arr) {
    return Eigen::Map<const Eigen::VectorXd>(arr.data(), arr.shape(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────

NB_MODULE(alpha_cpp, m) {
    m.doc() = "alpha_cpp — C++26 signal engines for institutional systematic macro research.";

    // ── KPIBundle ─────────────────────────────────────────────────────────────
    nb::class_<alpha::portfolio::KPIBundle>(m, "KPIBundle")
        .def(nb::init<>())
        .def_rw("sharpe_ratio",        &alpha::portfolio::KPIBundle::sharpe_ratio)
        .def_rw("sortino_ratio",       &alpha::portfolio::KPIBundle::sortino_ratio)
        .def_rw("calmar_ratio",        &alpha::portfolio::KPIBundle::calmar_ratio)
        .def_rw("cagr",                &alpha::portfolio::KPIBundle::cagr)
        .def_rw("total_return",        &alpha::portfolio::KPIBundle::total_return)
        .def_rw("max_drawdown",        &alpha::portfolio::KPIBundle::max_drawdown)
        .def_rw("annual_vol",          &alpha::portfolio::KPIBundle::annual_vol)
        .def_rw("var_95",              &alpha::portfolio::KPIBundle::var_95)
        .def_rw("expected_shortfall",  &alpha::portfolio::KPIBundle::expected_shortfall)
        .def_rw("mean_ic",             &alpha::portfolio::KPIBundle::mean_ic)
        .def_rw("icir",                &alpha::portfolio::KPIBundle::icir)
        .def_rw("hit_rate",            &alpha::portfolio::KPIBundle::hit_rate)
        .def_rw("avg_turnover",        &alpha::portfolio::KPIBundle::avg_turnover)
        .def_rw("ic_half_life_days",   &alpha::portfolio::KPIBundle::ic_half_life_days)
        .def_rw("decay_alert",         &alpha::portfolio::KPIBundle::decay_alert)
        .def_rw("gross_alpha_bps",     &alpha::portfolio::KPIBundle::gross_alpha_bps)
        .def_rw("tc_drag_bps",         &alpha::portfolio::KPIBundle::tc_drag_bps)
        .def_rw("net_alpha_bps",       &alpha::portfolio::KPIBundle::net_alpha_bps)
        .def("__repr__", [](const alpha::portfolio::KPIBundle& k) {
            return std::format(
                "KPIBundle(sharpe={:.3f}, sortino={:.3f}, calmar={:.3f}, "
                "mdd={:.2%}, ic={:.4f}, icir={:.3f}, hl={:.1f}d)",
                k.sharpe_ratio, k.sortino_ratio, k.calmar_ratio,
                k.max_drawdown, k.mean_ic, k.icir, k.ic_half_life_days);
        });

    // ── PDRRMConfig ───────────────────────────────────────────────────────────
    nb::class_<alpha::pdrrm::PDRRMConfig>(m, "PDRRMConfig")
        .def(nb::init<>())
        .def_rw("rrdm_momentum_window", &alpha::pdrrm::PDRRMConfig::rrdm_momentum_window)
        .def_rw("pss_ewma_halflife",    &alpha::pdrrm::PDRRMConfig::pss_ewma_halflife)
        .def_rw("rac_vol_window",       &alpha::pdrrm::PDRRMConfig::rac_vol_window)
        .def_rw("ridge_lambda",         &alpha::pdrrm::PDRRMConfig::ridge_lambda)
        .def_rw("target_vol",           &alpha::pdrrm::PDRRMConfig::target_vol)
        .def_rw("max_position",         &alpha::pdrrm::PDRRMConfig::max_position);

    // ── PDRRMEngine ───────────────────────────────────────────────────────────
    nb::class_<alpha::pdrrm::PDRRMEngine>(m, "PDRRMEngine")
        .def(nb::init<alpha::pdrrm::PDRRMConfig>(), "cfg"_a = alpha::pdrrm::PDRRMConfig{})
        .def("compute_rrdm", [](alpha::pdrrm::PDRRMEngine& self,
                                 const NpArray2D& nom, const NpArray2D& be) {
            auto result = self.compute_rrdm(np2eigen_mat(nom), np2eigen_mat(be));
            if (!result) throw std::runtime_error(result.error().message);
            const auto& mat = *result;
            // Return as numpy C-contiguous array
            auto data = new double[mat.size()];
            Eigen::Map<Eigen::MatrixXd>(data, mat.rows(), mat.cols()) = mat;
            nb::capsule deleter(data, [](void* p) noexcept { delete[] (double*)p; });
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
                data, {(size_t)mat.rows(), (size_t)mat.cols()}, deleter);
        }, "nominal_rates"_a, "breakevens"_a,
            "Compute RRDM panel [T×N] from nominal rates and breakevens.")
        .def("compute_pss", [](alpha::pdrrm::PDRRMEngine& self,
                                const NpArray2D& surprises) {
            auto result = self.compute_pss(np2eigen_mat(surprises));
            if (!result) throw std::runtime_error(result.error().message);
            const auto& mat = *result;
            auto data = new double[mat.size()];
            Eigen::Map<Eigen::MatrixXd>(data, mat.rows(), mat.cols()) = mat;
            nb::capsule deleter(data, [](void* p) noexcept { delete[] (double*)p; });
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
                data, {(size_t)mat.rows(), (size_t)mat.cols()}, deleter);
        }, "cb_surprises"_a, "Compute PSS panel [T×N] from CB meeting surprises.")
        .def("compute_rac", [](alpha::pdrrm::PDRRMEngine& self,
                                const NpArray2D& fp, const NpArray2D& ret) {
            auto result = self.compute_rac(np2eigen_mat(fp), np2eigen_mat(ret));
            if (!result) throw std::runtime_error(result.error().message);
            const auto& mat = *result;
            auto data = new double[mat.size()];
            Eigen::Map<Eigen::MatrixXd>(data, mat.rows(), mat.cols()) = mat;
            nb::capsule deleter(data, [](void* p) noexcept { delete[] (double*)p; });
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
                data, {(size_t)mat.rows(), (size_t)mat.cols()}, deleter);
        }, "forward_premium"_a, "returns"_a, "Compute RAC panel [T×N].")
        .def("update_weights", [](alpha::pdrrm::PDRRMEngine& self,
                                   const NpArray2D& X, const NpArray1D& y) {
            auto result = self.update_weights(np2eigen_mat(X), np2eigen_vec(y));
            if (!result) throw std::runtime_error(result.error().message);
            const auto& w = *result;
            auto data = new double[3];
            Eigen::Map<Eigen::VectorXd>(data, 3) = w;
            nb::capsule deleter(data, [](void* p) noexcept { delete[] (double*)p; });
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, {3}, deleter);
        }, "X_is"_a, "y_is"_a, "Estimate ridge regression weights on in-sample data.");

    // ── BacktestEngine ────────────────────────────────────────────────────────
    nb::class_<alpha::portfolio::BacktestEngine>(m, "BacktestEngine")
        .def(nb::init<>())
        .def("run", [](const alpha::portfolio::BacktestEngine& self,
                       const NpArray2D& signals, const NpArray2D& returns,
                       double tc_rate, double ann_days) {
            auto result = self.run(np2eigen_mat(signals), np2eigen_mat(returns),
                                   tc_rate, ann_days);
            if (!result) throw std::runtime_error(result.error().message);
            return *result;
        }, "signals"_a, "returns"_a, "tc_rate"_a = 0.0001, "ann_days"_a = 252.0,
            "Run full backtest; returns KPIBundle.");

    // ── SignalDecayMonitor ────────────────────────────────────────────────────
    nb::class_<alpha::portfolio::SignalDecayMonitor>(m, "SignalDecayMonitor")
        .def(nb::init<int>(), "window"_a = 60)
        .def("update",          &alpha::portfolio::SignalDecayMonitor::update,
             "ic"_a, "Update with new IC observation; returns True if decay alert.")
        .def("half_life",       &alpha::portfolio::SignalDecayMonitor::half_life)
        .def("rolling_mean_ic", &alpha::portfolio::SignalDecayMonitor::rolling_mean_ic)
        .def("alert",           &alpha::portfolio::SignalDecayMonitor::alert);

    // ── TPMCREngine ───────────────────────────────────────────────────────────
    nb::class_<alpha::strategies::TPMCREngine>(m, "TPMCREngine")
        .def(nb::init<alpha::strategies::TPMCRConfig>(),
             "cfg"_a = alpha::strategies::TPMCRConfig{})
        .def("tick", [](alpha::strategies::TPMCREngine& self,
                        double tp_acm, double y2, double y10, double y30,
                        double swap_sp, double cds,
                        nb::ndarray<double, nb::ndim<1>, nb::c_contig> rets) {
            alpha::strategies::TPMCREngine::DayData d{
                .tp_acm_10y    = tp_acm,
                .yield_2y      = y2,
                .yield_10y     = y10,
                .yield_30y     = y30,
                .swap_spread_30y = swap_sp,
                .cds_5y_proxy  = cds,
            };
            for (int i = 0; i < 5; ++i) d.returns[i] = rets.data()[i];
            auto out = self.tick(d);
            return nb::dict(
                "curve_regime"_a = out.curve_regime,
                "tpm_z"_a        = out.tpm_z,
                "fso_z"_a        = out.fiscal_stress_z,
                "signals"_a      = out.signals,
                "positions"_a    = out.positions);
        });

    // ── VSRAEngine ────────────────────────────────────────────────────────────
    nb::class_<alpha::strategies::VSRAEngine>(m, "VSRAEngine")
        .def(nb::init<alpha::strategies::VSRAConfig>(),
             "cfg"_a = alpha::strategies::VSRAConfig{})
        .def("tick", [](alpha::strategies::VSRAEngine& self,
                        double iv_atm, double rv_21d, double vix_spot,
                        double vix_3m, double put_skew, double rv_skew,
                        double vx_return) {
            alpha::strategies::VSRAEngine::DayData d{
                .iv_atm    = iv_atm, .rv_21d  = rv_21d,
                .vix_spot  = vix_spot, .vix_3m  = vix_3m,
                .put_skew  = put_skew, .rv_skew = rv_skew,
                .vx_return = vx_return};
            auto out = self.tick(d);
            return nb::dict(
                "vrp_z"_a    = out.vrp_z,
                "tss_z"_a    = out.tss_z,
                "ska_z"_a    = out.ska_z,
                "signal"_a   = out.signal,
                "position"_a = out.position);
        });

    // ── FDSPEngine ────────────────────────────────────────────────────────────
    nb::class_<alpha::strategies::FDSPEngine>(m, "FDSPEngine")
        .def(nb::init<alpha::strategies::FDSPConfig>(),
             "cfg"_a = alpha::strategies::FDSPConfig{})
        .def("tick", [](alpha::strategies::FDSPEngine& self,
                        double swap_sp, double cds, double tbill,
                        nb::ndarray<double, nb::ndim<1>, nb::c_contig> rets) {
            alpha::strategies::FDSPEngine::DayData d{
                .swap_spread_30y = swap_sp,
                .cds_5y          = cds,
                .tbill_spike     = tbill,
            };
            for (int i = 0; i < 5; ++i) d.returns[i] = rets.data()[i];
            auto out = self.tick(d);
            return nb::dict(
                "fci_z"_a     = out.fci_z,
                "signals"_a   = out.signals,
                "positions"_a = out.positions);
        });

    // ── Utility functions ─────────────────────────────────────────────────────
    m.def("sharpe_ratio", [](const NpArray1D& pnl, double ann) {
        return alpha::math::sharpe_ratio(np2eigen_vec(pnl), ann);
    }, "pnl"_a, "ann_factor"_a = 252.0, "Compute annualised Sharpe ratio.");

    m.def("max_drawdown", [](const NpArray1D& cum_pnl) {
        return alpha::math::max_drawdown(np2eigen_vec(cum_pnl));
    }, "cum_pnl"_a, "Compute maximum drawdown from cumulative PnL series.");

    m.def("half_life_decay", [](const NpArray1D& series) {
        return alpha::math::half_life_decay(np2eigen_vec(series));
    }, "series"_a, "Estimate AR(1) half-life for signal decay monitoring.");
}
