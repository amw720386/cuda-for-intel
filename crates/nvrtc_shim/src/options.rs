//! NVRTC option parsing and mapping to clang's CUDA front end.

#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    /// Requested virtual/real architecture (e.g. 75 for compute_75 / sm_75).
    pub arch: u32,
    pub std: String,
    pub default_device: bool,
    pub fast_math: bool,
    pub ftz: bool,
    pub fmad: bool,
    pub no_warnings: bool,
    /// Arguments forwarded verbatim to clang (-D, -U, -I, -include).
    pub clang_args: Vec<String>,
    /// Options accepted and deliberately ignored, for the log.
    pub ignored: Vec<String>,
}

impl Default for Parsed {
    fn default() -> Self {
        Self {
            arch: 52,
            std: "c++17".into(),
            default_device: false,
            fast_math: false,
            ftz: false,
            fmad: true,
            no_warnings: false,
            clang_args: Vec::new(),
            ignored: Vec::new(),
        }
    }
}

/// (spellings, takes value). Value may follow `=` or be the next argument.
const TABLE: &[(&[&str], bool)] = &[
    (&["--gpu-architecture", "-arch"], true),
    (&["--std"], true),
    (&["--define-macro", "-D"], true),
    (&["--undefine-macro", "-U"], true),
    (&["--include-path", "-I"], true),
    (&["--pre-include", "-include"], true),
    (&["--fmad", "-fmad"], true),
    (&["--ftz", "-ftz"], true),
    (&["--prec-sqrt", "-prec-sqrt"], true),
    (&["--prec-div", "-prec-div"], true),
    (&["--maxrregcount", "-maxrregcount"], true),
    (&["--ptxas-options", "-Xptxas"], true),
    (&["--dopt", "-dopt"], true),
    (&["--device-as-default-execution-space", "-default-device"], false),
    (&["--use_fast_math", "-use_fast_math"], false),
    (&["--relocatable-device-code", "-rdc"], true),
    (&["--device-c", "-dc"], false),
    (&["--device-w", "-dw"], false),
    (&["--extensible-whole-program", "-ewp"], false),
    (&["--device-debug", "-G"], false),
    (&["--generate-line-info", "-lineinfo"], false),
    (&["--dlink-time-opt", "-dlto"], false),
    (&["--gen-opt-lto"], false),
    (&["--split-compile"], true),
    (&["--extra-device-vectorization"], false),
    (&["--modify-stack-limit"], true),
    (&["--disable-warnings", "-w"], false),
    (&["--restrict", "-restrict"], false),
    (&["--device-int128"], false),
    (&["--device-float128"], false),
    (&["--optimization-info", "-opt-info"], true),
    (&["--display-error-number", "-err-no"], false),
    (&["--no-display-error-number", "-no-err-no"], false),
    (&["--diag-error", "-diag-error"], true),
    (&["--diag-suppress", "-diag-suppress"], true),
    (&["--diag-warn", "-diag-warn"], true),
    (&["--brief-diagnostics", "-brief-diag"], true),
    (&["--no-source-include", "-no-source-include"], false),
    (&["--builtin-move-forward"], true),
    (&["--builtin-initializer-list"], true),
    (&["--minimal"], false),
    (&["--time"], true),
    (&["--fdevice-syntax-only"], false),
    (&["--jump-table-density"], true),
    (&["--Wreorder", "-Wreorder"], false),
    (&["--Werror", "-Werror"], true),
];

fn parse_arch(v: &str) -> Result<u32, String> {
    let digits = v
        .strip_prefix("compute_")
        .or_else(|| v.strip_prefix("sm_"))
        .or_else(|| v.strip_prefix("lto_"))
        .ok_or_else(|| format!("unsupported architecture '{v}'"))?;
    let digits: String = digits.chars().take_while(char::is_ascii_digit).collect();
    digits.parse::<u32>().map_err(|_| format!("unsupported architecture '{v}'"))
}

fn truthy(v: &str) -> Result<bool, String> {
    match v {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(format!("invalid boolean '{other}'")),
    }
}

pub fn parse<S: AsRef<str>>(opts: &[S]) -> Result<Parsed, String> {
    let mut out = Parsed::default();
    let mut i = 0;
    while i < opts.len() {
        let raw = opts[i].as_ref();
        i += 1;
        // Split `--name=value`; short -D/-I/-U also accept an attached value (`-DFOO=1`).
        let (name, mut value): (&str, Option<String>) = match raw.split_once('=') {
            Some((n, v)) if n.starts_with('-') && TABLE.iter().any(|(s, _)| s.contains(&n)) => (n, Some(v.to_owned())),
            _ => (raw, None),
        };
        let mut entry = TABLE.iter().find(|(spellings, _)| spellings.contains(&name));
        if entry.is_none() {
            for short in ["-D", "-U", "-I"] {
                if let Some(rest) = raw.strip_prefix(short).filter(|r| !r.is_empty()) {
                    entry = TABLE.iter().find(|(s, _)| s.contains(&short));
                    value = Some(rest.to_owned());
                    break;
                }
            }
        }
        let Some(&(spellings, takes)) = entry else {
            return Err(format!("unrecognized option '{raw}'"));
        };
        let canon = spellings[0];
        if takes && value.is_none() {
            // `--rdc` style flags accept an optional bool; the rest require an argument.
            if canon == "--relocatable-device-code" {
                value = Some("true".into());
            } else {
                let next = opts.get(i).map(|s| s.as_ref().to_owned());
                match next {
                    Some(v) => {
                        value = Some(v);
                        i += 1;
                    }
                    None => return Err(format!("option '{raw}' requires a value")),
                }
            }
        }
        let v = value.unwrap_or_default();
        match canon {
            "--gpu-architecture" => out.arch = parse_arch(&v)?,
            "--std" => {
                out.std = match v.as_str() {
                    "c++03" | "c++11" => "c++11".into(),
                    "c++14" | "c++17" | "c++20" => v.clone(),
                    other => return Err(format!("unsupported standard '{other}'")),
                }
            }
            "--define-macro" => out.clang_args.push(format!("-D{v}")),
            "--undefine-macro" => out.clang_args.push(format!("-U{v}")),
            "--include-path" => out.clang_args.push(format!("-I{v}")),
            "--pre-include" => {
                out.clang_args.push("-include".into());
                out.clang_args.push(v);
            }
            "--fmad" => out.fmad = truthy(&v)?,
            "--ftz" => out.ftz = truthy(&v)?,
            "--use_fast_math" => {
                out.fast_math = true;
                out.ftz = true;
            }
            "--device-as-default-execution-space" => out.default_device = true,
            "--disable-warnings" => out.no_warnings = true,
            _ => out.ignored.push(raw.to_owned()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_common_options() {
        let p = parse(&["--gpu-architecture=compute_80", "-DFOO=1", "-I", "inc", "--std=c++14", "--fmad=false", "-default-device"]).unwrap();
        assert_eq!(p.arch, 80);
        assert_eq!(p.std, "c++14");
        assert!(!p.fmad && p.default_device);
        assert_eq!(p.clang_args, ["-DFOO=1", "-Iinc"]);
    }

    #[test]
    fn rejects_unknown() {
        assert!(parse(&["--bogus"]).is_err());
        assert!(parse(&["--gpu-architecture=sm_x"]).is_err());
    }

    #[test]
    fn ignores_rdc() {
        let p = parse(&["--relocatable-device-code=true", "-rdc=true", "-arch=sm_90a", "--maxrregcount=64"]).unwrap();
        assert_eq!(p.arch, 90);
        assert_eq!(p.ignored.len(), 3);
    }
}
