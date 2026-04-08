# third_party/nanobind.BUILD
cc_library(
    name = "nanobind",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob(["include/nanobind/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    copts = ["-std=c++26", "-O2"],
)
