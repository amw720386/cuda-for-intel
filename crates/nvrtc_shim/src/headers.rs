//! Built-in header set written to a content-addressed temporary directory for clang.

pub const BUILTINS: &str = "nvrtc_shim_builtins.h";

/// (virtual path, contents). Files that merely forward to another header are listed inline.
pub fn files() -> Vec<(&'static str, &'static str)> {
    vec![
        (BUILTINS, include_str!("../include/nvrtc_shim_builtins.h")),
        ("cuda_fp16.h", include_str!("../include/cuda_fp16.h")),
        ("cuda_bf16.h", include_str!("../include/cuda_bf16.h")),
        ("cooperative_groups.h", include_str!("../include/cooperative_groups.h")),
        ("cooperative_groups/reduce.h", include_str!("../include/cooperative_groups/reduce.h")),
        ("cuda_runtime.h", include_str!("../include/cuda_runtime.h")),
        ("stdint.h", include_str!("../include/stdint.h")),
        ("limits.h", include_str!("../include/limits.h")),
        ("float.h", include_str!("../include/float.h")),
        ("math.h", include_str!("../include/math.h")),
        ("stdio.h", include_str!("../include/stdio.h")),
        ("cuda_fp16.hpp", "#include \"cuda_fp16.h\"\n"),
        ("cuda_runtime_api.h", "#include \"cuda_runtime.h\"\n"),
        ("device_launch_parameters.h", "#include \"cuda_runtime.h\"\n"),
        ("stddef.h", "#include \"nvrtc_shim_builtins.h\"\n"),
        ("cstddef", "#include \"stddef.h\"\nnamespace std { using ::size_t; using ::ptrdiff_t; }\n"),
        ("cstdint", "#include \"stdint.h\"\nnamespace std { using ::int8_t; using ::uint8_t; using ::int16_t; using ::uint16_t; using ::int32_t; using ::uint32_t; using ::int64_t; using ::uint64_t; }\n"),
        ("climits", "#include \"limits.h\"\n"),
        ("cfloat", "#include \"float.h\"\n"),
        ("cmath", "#include \"math.h\"\nnamespace std { using ::sqrt; using ::fabs; using ::floor; using ::ceil; using ::exp; using ::log; using ::sin; using ::cos; using ::tanh; using ::pow; using ::min; using ::max; using ::abs; using ::fma; using ::erf; using ::isnan; using ::isinf; using ::isfinite; }\n"),
        ("cstdio", "#include \"stdio.h\"\n"),
        ("assert.h", "#include \"nvrtc_shim_builtins.h\"\n"),
    ]
}
