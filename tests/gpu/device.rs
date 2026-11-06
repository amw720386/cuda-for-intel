pub fn ordinal() -> i32 {
    match std::env::var("CUDA_FOR_INTEL_DEVICE") {
        Ok(value) => value
            .parse::<i32>()
            .ok()
            .filter(|&ordinal| ordinal >= 0)
            .expect("CUDA_FOR_INTEL_DEVICE must be a nonnegative Level Zero device ordinal"),
        Err(std::env::VarError::NotPresent) => 0,
        Err(error) => panic!("CUDA_FOR_INTEL_DEVICE: {error}"),
    }
}
