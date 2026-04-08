"""WORKSPACE — Bazel workspace for alpha-research-2026.

Monorepo: C++26 signal engines + Python 3.13 research pipeline.
Bridges via nanobind. Build gateway for CI, Docker, and local runs.

Author: Alpha Research Pod — 2026
"""

workspace(name = "alpha_research_2026")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# ── Bazel Skylib ───────────────────────────────────────────────────────────────
http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

# ── Rules Python ──────────────────────────────────────────────────────────────
http_archive(
    name = "rules_python",
    sha256 = "9c6e26911a79fbf510a8f06d8eedb40f412023cf7fa6d1461def27116bff022c",
    strip_prefix = "rules_python-0.40.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.40.0/rules_python-0.40.0.tar.gz",
)
load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")
py_repositories()
python_register_toolchains(
    name = "python3_13",
    python_version = "3.13",
)
load("@python3_13//:defs.bzl", "interpreter")

# ── Google Test ────────────────────────────────────────────────────────────────
http_archive(
    name = "googletest",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

# ── Eigen3 ────────────────────────────────────────────────────────────────────
http_archive(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-3.4.0",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
)

# ── nanobind ─────────────────────────────────────────────────────────────────
http_archive(
    name = "nanobind",
    build_file = "//third_party:nanobind.BUILD",
    sha256 = "a2e5ad5c2f4b4a86c8d2c4e3c3b2a15e29abf1a32e55e5d9c7a4b9e2a36a2b14",
    strip_prefix = "nanobind-2.4.0",
    urls = ["https://github.com/wjakob/nanobind/archive/refs/tags/v2.4.0.tar.gz"],
)

# ── pybind11 (fallback, used by nanobind transitively) ────────────────────────
http_archive(
    name = "pybind11_bazel",
    sha256 = "fec6281e4109115c5157ca720b8fe20c8f655f773172290b03f57353c11869c2",
    strip_prefix = "pybind11_bazel-2.12.0",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/v2.12.0.tar.gz"],
)
