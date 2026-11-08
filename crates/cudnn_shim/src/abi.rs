//! cuDNN ABI constants, checked against the cuDNN v8/v9 headers.
//! Status codes differ between major versions: the default build uses the v9
//! numbering (`cudnn64_9.dll`), `--features v8` the v8 numbering (`cudnn64_8.dll`).
#![allow(dead_code)]

pub type Status = i32;

pub const SUCCESS: Status = 0;

#[cfg(not(feature = "v8"))]
mod st {
    pub const NOT_INITIALIZED: i32 = 1001;
    pub const BAD_PARAM: i32 = 2000;
    pub const BAD_PARAM_NULL_POINTER: i32 = 2002;
    pub const BAD_PARAM_NOT_FINALIZED: i32 = 2004;
    pub const BAD_PARAM_OUT_OF_BOUND: i32 = 2005;
    pub const BAD_PARAM_SIZE_INSUFFICIENT: i32 = 2006;
    pub const BAD_PARAM_SHAPE_MISMATCH: i32 = 2008;
    pub const BAD_PARAM_ATTRIBUTE_TYPE: i32 = 2010;
    pub const BAD_PARAM_DESCRIPTOR_TYPE: i32 = 2012;
    pub const INVALID_VALUE: i32 = 2001;
    pub const NOT_SUPPORTED: i32 = 3000;
    pub const NOT_SUPPORTED_GRAPH_PATTERN: i32 = 3001;
    pub const NOT_SUPPORTED_SHAPE: i32 = 3002;
    pub const NOT_SUPPORTED_DATA_TYPE: i32 = 3003;
    pub const NOT_SUPPORTED_LAYOUT: i32 = 3004;
    pub const INTERNAL_ERROR: i32 = 4000;
    pub const ALLOC_FAILED: i32 = 4003;
    pub const EXECUTION_FAILED: i32 = 5000;
    pub const VERSION: usize = 90701;
    pub const MAJOR: i32 = 9;
    pub const MINOR: i32 = 7;
    pub const PATCH: i32 = 1;
}

#[cfg(feature = "v8")]
mod st {
    pub const NOT_INITIALIZED: i32 = 1;
    pub const BAD_PARAM: i32 = 3;
    pub const BAD_PARAM_NULL_POINTER: i32 = 3;
    pub const BAD_PARAM_NOT_FINALIZED: i32 = 3;
    pub const BAD_PARAM_OUT_OF_BOUND: i32 = 3;
    pub const BAD_PARAM_SIZE_INSUFFICIENT: i32 = 3;
    pub const BAD_PARAM_SHAPE_MISMATCH: i32 = 3;
    pub const BAD_PARAM_ATTRIBUTE_TYPE: i32 = 3;
    pub const BAD_PARAM_DESCRIPTOR_TYPE: i32 = 3;
    pub const INVALID_VALUE: i32 = 5;
    pub const NOT_SUPPORTED: i32 = 9;
    pub const NOT_SUPPORTED_GRAPH_PATTERN: i32 = 9;
    pub const NOT_SUPPORTED_SHAPE: i32 = 9;
    pub const NOT_SUPPORTED_DATA_TYPE: i32 = 9;
    pub const NOT_SUPPORTED_LAYOUT: i32 = 9;
    pub const INTERNAL_ERROR: i32 = 4;
    pub const ALLOC_FAILED: i32 = 2;
    pub const EXECUTION_FAILED: i32 = 8;
    pub const VERSION: usize = 8907;
    pub const MAJOR: i32 = 8;
    pub const MINOR: i32 = 9;
    pub const PATCH: i32 = 7;
}
pub use st::*;

/// Name for `cudnnGetErrorString` (NUL terminated).
pub fn status_name(s: Status) -> &'static [u8] {
    match s {
        SUCCESS => b"CUDNN_STATUS_SUCCESS\0",
        NOT_INITIALIZED => b"CUDNN_STATUS_NOT_INITIALIZED\0",
        ALLOC_FAILED => b"CUDNN_STATUS_ALLOC_FAILED\0",
        INTERNAL_ERROR => b"CUDNN_STATUS_INTERNAL_ERROR\0",
        EXECUTION_FAILED => b"CUDNN_STATUS_EXECUTION_FAILED\0",
        NOT_SUPPORTED => b"CUDNN_STATUS_NOT_SUPPORTED\0",
        NOT_SUPPORTED_GRAPH_PATTERN => b"CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN\0",
        NOT_SUPPORTED_SHAPE => b"CUDNN_STATUS_NOT_SUPPORTED_SHAPE\0",
        NOT_SUPPORTED_DATA_TYPE => b"CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE\0",
        NOT_SUPPORTED_LAYOUT => b"CUDNN_STATUS_NOT_SUPPORTED_LAYOUT\0",
        BAD_PARAM => b"CUDNN_STATUS_BAD_PARAM\0",
        BAD_PARAM_NULL_POINTER => b"CUDNN_STATUS_BAD_PARAM_NULL_POINTER\0",
        BAD_PARAM_NOT_FINALIZED => b"CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED\0",
        BAD_PARAM_OUT_OF_BOUND => b"CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND\0",
        BAD_PARAM_SIZE_INSUFFICIENT => b"CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT\0",
        BAD_PARAM_SHAPE_MISMATCH => b"CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH\0",
        BAD_PARAM_ATTRIBUTE_TYPE => b"CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE\0",
        BAD_PARAM_DESCRIPTOR_TYPE => b"CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE\0",
        INVALID_VALUE => b"CUDNN_STATUS_INVALID_VALUE\0",
        _ => b"CUDNN_STATUS_UNKNOWN\0",
    }
}

// cudnnDataType_t
pub const DT_FLOAT: i32 = 0;
pub const DT_DOUBLE: i32 = 1;
pub const DT_HALF: i32 = 2;
pub const DT_INT8: i32 = 3;
pub const DT_INT32: i32 = 4;
pub const DT_BFLOAT16: i32 = 9;
pub const DT_INT64: i32 = 10;
pub const DT_BOOLEAN: i32 = 11;
// cudnnTensorFormat_t
pub const NCHW: i32 = 0;
pub const NHWC: i32 = 1;
pub const NCHW_VECT_C: i32 = 2;
// cudnnConvolutionMode_t
pub const CONVOLUTION: i32 = 0;
pub const CROSS_CORRELATION: i32 = 1;
// cudnnMathType_t
pub const DEFAULT_MATH: i32 = 0;
pub const FMA_MATH: i32 = 3;
// cudnnBatchNormMode_t
pub const BN_PER_ACTIVATION: i32 = 0;
pub const BN_SPATIAL: i32 = 1;
pub const BN_SPATIAL_PERSISTENT: i32 = 2;
// cudnnActivationMode_t
pub const ACT_SIGMOID: i32 = 0;
pub const ACT_RELU: i32 = 1;
pub const ACT_TANH: i32 = 2;
pub const ACT_CLIPPED_RELU: i32 = 3;
pub const ACT_ELU: i32 = 4;
pub const ACT_IDENTITY: i32 = 5;
pub const ACT_SWISH: i32 = 6;
// cudnnPoolingMode_t
pub const POOL_MAX: i32 = 0;
pub const POOL_MAX_DETERMINISTIC: i32 = 3;
// cudnnDeterminism_t
pub const DETERMINISTIC: i32 = 1;

// Algorithms (the shim implements one direct kernel per direction and reports it as the first algo).
pub const FWD_ALGO_IMPLICIT_GEMM: i32 = 0;
pub const BWD_DATA_ALGO_0: i32 = 0;
pub const BWD_FILTER_ALGO_0: i32 = 0;
pub const FWD_ALGO_COUNT: i32 = 8;
pub const BWD_DATA_ALGO_COUNT: i32 = 6;
pub const BWD_FILTER_ALGO_COUNT: i32 = 7;

/// `cudnnConvolution{Fwd,BwdData,BwdFilter}AlgoPerf_t` (all three share this layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlgoPerf {
    pub algo: i32,
    pub status: i32,
    pub time: f32,
    pub memory: usize,
    pub determinism: i32,
    pub math_type: i32,
    pub reserved: [i32; 3],
}

// Backend descriptor types (cudnnBackendDescriptorType_t)
pub const BD_CONVOLUTION: i32 = 1;
pub const BD_ENGINE: i32 = 2;
pub const BD_ENGINECFG: i32 = 3;
pub const BD_ENGINEHEUR: i32 = 4;
pub const BD_EXECUTION_PLAN: i32 = 5;
pub const BD_KNOB_CHOICE: i32 = 7;
pub const BD_OP_CONV_FWD: i32 = 10;
pub const BD_OP_CONV_BWD_FILTER: i32 = 11;
pub const BD_OP_CONV_BWD_DATA: i32 = 12;
pub const BD_OPERATIONGRAPH: i32 = 15;
pub const BD_VARIANT_PACK: i32 = 16;
pub const BD_TENSOR: i32 = 17;
pub const BD_MAX: i32 = 41;

// cudnnBackendAttributeType_t
pub const AT_HANDLE: i32 = 0;
pub const AT_DATA_TYPE: i32 = 1;
pub const AT_BOOLEAN: i32 = 2;
pub const AT_INT64: i32 = 3;
pub const AT_FLOAT: i32 = 4;
pub const AT_DOUBLE: i32 = 5;
pub const AT_VOID_PTR: i32 = 6;
pub const AT_CONVOLUTION_MODE: i32 = 7;
pub const AT_BACKEND_DESCRIPTOR: i32 = 15;

// cudnnBackendAttributeName_t (subset)
pub const A_CONV_COMP_TYPE: i32 = 100;
pub const A_CONV_MODE: i32 = 101;
pub const A_CONV_DILATIONS: i32 = 102;
pub const A_CONV_FILTER_STRIDES: i32 = 103;
pub const A_CONV_POST_PADDINGS: i32 = 104;
pub const A_CONV_PRE_PADDINGS: i32 = 105;
pub const A_CONV_SPATIAL_DIMS: i32 = 106;
pub const A_HEUR_MODE: i32 = 200;
pub const A_HEUR_GRAPH: i32 = 201;
pub const A_HEUR_RESULTS: i32 = 202;
pub const A_CFG_ENGINE: i32 = 300;
pub const A_CFG_KNOBS: i32 = 302;
pub const A_CFG_WORKSPACE: i32 = 303;
pub const A_PLAN_HANDLE: i32 = 400;
pub const A_PLAN_CFG: i32 = 401;
pub const A_PLAN_WORKSPACE: i32 = 402;
pub const A_FWD_ALPHA: i32 = 700;
pub const A_FWD_BETA: i32 = 701;
pub const A_FWD_CONV: i32 = 702;
pub const A_FWD_W: i32 = 703;
pub const A_FWD_X: i32 = 704;
pub const A_FWD_Y: i32 = 705;
pub const A_BWDD_ALPHA: i32 = 706;
pub const A_BWDD_BETA: i32 = 707;
pub const A_BWDD_CONV: i32 = 708;
pub const A_BWDD_W: i32 = 709;
pub const A_BWDD_DX: i32 = 710;
pub const A_BWDD_DY: i32 = 711;
pub const A_BWDF_ALPHA: i32 = 712;
pub const A_BWDF_BETA: i32 = 713;
pub const A_BWDF_CONV: i32 = 714;
pub const A_BWDF_DW: i32 = 715;
pub const A_BWDF_X: i32 = 716;
pub const A_BWDF_DY: i32 = 717;
pub const A_GRAPH_HANDLE: i32 = 800;
pub const A_GRAPH_OPS: i32 = 801;
pub const A_GRAPH_ENGINE_COUNT: i32 = 802;
pub const A_TENSOR_ALIGN: i32 = 900;
pub const A_TENSOR_DTYPE: i32 = 901;
pub const A_TENSOR_DIMS: i32 = 902;
pub const A_TENSOR_STRIDES: i32 = 903;
pub const A_TENSOR_UID: i32 = 906;
pub const A_TENSOR_VIRTUAL: i32 = 907;
pub const A_VP_UIDS: i32 = 1000;
pub const A_VP_PTRS: i32 = 1001;
pub const A_VP_WORKSPACE: i32 = 1003;
pub const A_ENGINE_GRAPH: i32 = 1300;
pub const A_ENGINE_INDEX: i32 = 1301;
