//! Pre-built SPIR-V modules inside the `cuModuleLoadData` text channel.
//!
//! `cuModuleLoadData` only carries NUL-terminated text, but some libraries (the
//! cuBLAS shim's native XMX GEMM) generate SPIR-V directly because PTX cannot
//! express `OpSubgroupMatrixMultiplyAccumulateINTEL`. Such a module is wrapped
//! in an ASCII container that [`parse`] unwraps into a [`Translation`]; the
//! rest of the loader (module build, launch argument layout) is unchanged.
//!
//! ```text
//! //CFI-SPIRV 1
//! //ENTRY <name> <u64|u32|f32|u16|u8>,...   (one line per kernel)
//! //THREADS <max threads per workgroup>     (optional, applies to the last ENTRY)
//! //WORDS
//! 07230203 00010300 ...                      (SPIR-V words, hex, whitespace separated)
//! ```
use super::{Error, KernelMetadata, Param, Translation};

pub(crate) const MAGIC: &str = "//CFI-SPIRV 1";

fn param(token: &str) -> Result<Param, Error> {
    Ok(match token {
        "u8" => Param::U8,
        "u16" => Param::U16,
        "u32" => Param::U32,
        "f32" => Param::F32,
        "u64" => Param::U64,
        other => return Err(Error::new(format!("unsupported SPIR-V container parameter {other}"))),
    })
}

/// `None` when `source` is ordinary PTX.
pub(crate) fn parse(source: &str) -> Option<Result<Translation, Error>> {
    source.starts_with(MAGIC).then(|| parse_container(source))
}

fn parse_container(source: &str) -> Result<Translation, Error> {
    let mut kernels: Vec<KernelMetadata> = Vec::new();
    let mut words = Vec::new();
    let mut in_words = false;
    for line in source.lines().skip(1) {
        let line = line.trim();
        if in_words {
            for token in line.split_whitespace() {
                words.push(
                    u32::from_str_radix(token, 16)
                        .map_err(|_| Error::new("malformed SPIR-V word in container"))?,
                );
            }
        } else if let Some(rest) = line.strip_prefix("//ENTRY ") {
            let (name, params) = rest.split_once(' ').unwrap_or((rest, ""));
            let params = params
                .split(',')
                .filter(|t| !t.is_empty())
                .map(param)
                .collect::<Result<Vec<_>, _>>()?;
            kernels.push(KernelMetadata {
                name: name.to_owned(),
                params,
                shared_bytes: 0,
                max_threads: None,
                device_atomics: false,
                fp32: false,
                fp32_fma: false,
                // Precompiled SPIR-V: no PTX to analyse.
                direct_access: false,
            });
        } else if let Some(rest) = line.strip_prefix("//THREADS ") {
            let threads = rest.trim().parse().map_err(|_| Error::new("bad THREADS line"))?;
            kernels
                .last_mut()
                .ok_or_else(|| Error::new("THREADS before ENTRY"))?
                .max_threads = Some(threads);
        } else if line == "//WORDS" {
            in_words = true;
        }
    }
    if kernels.is_empty() || words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::new("SPIR-V container without entries or with a bad module"));
    }
    Ok(Translation { words, kernels, globals: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_ptx_is_not_a_container() {
        assert!(parse(".version 8.8\n").is_none());
    }

    #[test]
    fn container_round_trips() {
        let text = "//CFI-SPIRV 1\n//ENTRY k u64,u32,f32\n//THREADS 16\n//WORDS\n07230203 00010300 00000000 00000005 00000000\n";
        let t = parse(text).unwrap().unwrap();
        assert_eq!(t.kernels[0].name, "k");
        assert_eq!(t.kernels[0].params, vec![Param::U64, Param::U32, Param::F32]);
        assert_eq!(t.kernels[0].max_threads, Some(16));
        assert_eq!(t.words.len(), 5);
        assert!(parse("//CFI-SPIRV 1\n//ENTRY k u64\n//WORDS\n1 2 3 4 5\n").unwrap().is_err());
    }
}
