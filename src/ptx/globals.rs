//! Module-scope PTX variables: `.global`, `.const` and `.shared` declarations.
//!
//! Device variables (`.global` and `.const`) become SPIR-V `CrossWorkgroup`
//! module variables with an exported linkage name. Level Zero then resolves
//! their device address with `zeModuleGetGlobalPointer`, so address-of-global
//! relocation is the driver's job and kernels need no hidden arguments.
//! `.const` data is placed in ordinary device memory: PTX kernels can only read
//! it, while `cuMemcpyHtoD` on the address from `cuModuleGetGlobal` may write it.
//! Module-scope `.shared` arrays are folded into each kernel that mentions them.
use super::Error;
use std::collections::HashSet;

/// Largest initializer the translator will embed in SPIR-V.
const MAX_INITIALIZER_BYTES: usize = 32 << 20;
/// Largest uninitialized or zero-filled variable.
const MAX_VARIABLE_BYTES: usize = 1 << 30;

/// One device-resident module variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GlobalVar {
    pub(crate) name: String,
    /// Declared with `.const`.
    pub(crate) constant: bool,
    /// Alignment in bytes, at least four.
    pub(crate) align: u32,
    /// Exact size of the PTX object in bytes.
    pub(crate) size: u32,
    /// `size` bytes of little-endian initial data. `None` means all zero.
    pub(crate) init: Option<Vec<u8>>,
}

/// A module-scope `.shared` array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SharedVar<'a> {
    pub(super) name: &'a str,
    /// Size rounded up to a multiple of four bytes.
    pub(super) size: u32,
    pub(super) align: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Declared<'a> {
    Device(GlobalVar),
    Shared(SharedVar<'a>),
    /// `.extern .shared`: the dynamic shared block, handled by the kernel parser.
    ExternShared,
    /// Declarations with no representation here (extern symbols, textures, ...).
    Ignored,
}

/// Declarations collected while walking the module.
#[derive(Default)]
pub(super) struct ModuleVars<'a> {
    pub(super) device: Vec<GlobalVar>,
    pub(super) shared: Vec<SharedVar<'a>>,
    device_names: HashSet<&'a str>,
}

impl<'a> ModuleVars<'a> {
    pub(super) fn contains_device(&self, name: &str) -> bool {
        self.device_names.contains(name)
    }

    pub(super) fn record(&mut self, tokens: &[&'a str]) {
        // An unsupported declaration only breaks kernels that use it.
        match declaration(tokens) {
            Ok(Declared::Device(var)) => {
                if let Some(&name) = tokens.iter().find(|&&token| token == var.name) {
                    if self.device_names.insert(name) {
                        self.device.push(var);
                    }
                }
            }
            Ok(Declared::Shared(var)) => {
                if !self.shared.iter().any(|known| known.name == var.name) {
                    self.shared.push(var);
                }
            }
            _ => {}
        }
    }
}

fn element_bytes(ty: &str) -> Option<u32> {
    Some(match ty {
        ".b8" | ".u8" | ".s8" => 1,
        ".b16" | ".u16" | ".s16" | ".f16" => 2,
        ".b32" | ".u32" | ".s32" | ".f32" => 4,
        ".b64" | ".u64" | ".s64" | ".f64" => 8,
        _ => return None,
    })
}

fn bad(message: &str) -> Error {
    Error::new(message)
}

fn parse_int(token: &str) -> Option<i128> {
    let (negative, digits) = match token.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, token.strip_prefix('+').unwrap_or(token)),
    };
    let magnitude = if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        i128::from_str_radix(hex, 16).ok()?
    } else if digits.len() > 1 && digits.starts_with('0') && digits.bytes().all(|b| b.is_ascii_digit()) {
        i128::from_str_radix(&digits[1..], 8).ok()?
    } else {
        digits.parse::<i128>().ok()?
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// Little-endian bytes of one initializer element.
fn encode(ty: &str, token: &str, out: &mut Vec<u8>) -> Result<(), Error> {
    let width = element_bytes(ty).ok_or_else(|| bad("unsupported PTX initializer type"))? as usize;
    let bits: u64 = match ty {
        ".f32" | ".f64" => {
            if let Some(hex) = token.strip_prefix("0f").or_else(|| token.strip_prefix("0F")) {
                if ty == ".f64" {
                    return Err(bad("PTX f64 initializer must use 0d"));
                }
                u32::from_str_radix(hex, 16)
                    .map(u64::from)
                    .map_err(|_| bad("invalid PTX 0f initializer"))?
            } else if let Some(hex) = token.strip_prefix("0d").or_else(|| token.strip_prefix("0D")) {
                if ty == ".f32" {
                    return Err(bad("PTX f32 initializer must use 0f"));
                }
                u64::from_str_radix(hex, 16).map_err(|_| bad("invalid PTX 0d initializer"))?
            } else if ty == ".f32" {
                token
                    .parse::<f32>()
                    .map(|value| u64::from(value.to_bits()))
                    .map_err(|_| bad("invalid PTX f32 initializer"))?
            } else {
                token
                    .parse::<f64>()
                    .map(f64::to_bits)
                    .map_err(|_| bad("invalid PTX f64 initializer"))?
            }
        }
        _ => {
            if token.contains(['.', '%']) || token.starts_with(|c: char| c.is_alphabetic() || c == '_' || c == '$') {
                return Err(bad("unsupported PTX symbolic initializer"));
            }
            let value = parse_int(token).ok_or_else(|| bad("invalid PTX integer initializer"))?;
            value as u64
        }
    };
    out.extend_from_slice(&bits.to_le_bytes()[..width]);
    Ok(())
}

/// Parse one declaration statement (without its terminating semicolon).
pub(super) fn declaration<'a>(tokens: &[&'a str]) -> Result<Declared<'a>, Error> {
    let mut at = 0;
    let mut external = false;
    let mut storage = None;
    let mut align = None;
    while let Some(&token) = tokens.get(at) {
        match token {
            ".visible" | ".weak" | ".common" => {}
            ".extern" => external = true,
            ".global" | ".const" | ".shared" if storage.is_none() => storage = Some(token),
            ".align" => {
                let value = tokens
                    .get(at + 1)
                    .and_then(|value| value.parse::<u32>().ok())
                    .filter(|value| value.is_power_of_two())
                    .ok_or_else(|| bad("invalid PTX variable alignment"))?;
                align = Some(value);
                at += 1;
            }
            _ => break,
        }
        at += 1;
    }
    let Some(storage) = storage else {
        return Ok(Declared::Ignored);
    };
    if storage == ".shared" && external {
        return Ok(Declared::ExternShared);
    }
    if external {
        return Ok(Declared::Ignored);
    }
    let ty = *tokens.get(at).ok_or_else(|| bad("truncated PTX variable"))?;
    // `.attribute`, vector types and opaque handles are not supported variables.
    let Some(elem) = element_bytes(ty) else {
        return Ok(Declared::Ignored);
    };
    let name = *tokens
        .get(at + 1)
        .ok_or_else(|| bad("truncated PTX variable"))?;
    if name.starts_with('%') || name.starts_with('.') || name == "=" || name == "[" {
        return Err(bad("invalid PTX variable name"));
    }
    at += 2;
    // Array dimensions. An empty `[]` takes its length from the initializer.
    let mut count: Option<u64> = Some(1);
    let mut unsized_array = false;
    while tokens.get(at) == Some(&"[") {
        let size = tokens.get(at + 1).ok_or_else(|| bad("truncated PTX array"))?;
        if *size == "]" {
            unsized_array = true;
            at += 2;
            continue;
        }
        let n = size
            .parse::<u64>()
            .ok()
            .filter(|&n| n != 0)
            .ok_or_else(|| bad("invalid PTX array length"))?;
        if tokens.get(at + 2) != Some(&"]") {
            return Err(bad("invalid PTX array declaration"));
        }
        count = count.and_then(|count| count.checked_mul(n));
        at += 3;
    }
    let mut init: Option<Vec<u8>> = None;
    match tokens.get(at) {
        None => {}
        Some(&"=") => {
            let mut bytes = Vec::new();
            for &token in &tokens[at + 1..] {
                if matches!(token, "{" | "}" | ",") {
                    continue;
                }
                encode(ty, token, &mut bytes)?;
                if bytes.len() > MAX_INITIALIZER_BYTES {
                    return Err(bad("PTX initializer is too large"));
                }
            }
            init = Some(bytes);
        }
        Some(_) => return Err(bad("unsupported PTX variable declaration")),
    }
    let elems = if unsized_array {
        let given = init.as_ref().map_or(0, |bytes| bytes.len() as u64 / u64::from(elem));
        if given == 0 { return Err(bad("unsized PTX array without initializer")); }
        given
    } else {
        count.ok_or_else(|| bad("PTX variable size overflows"))?
    };
    let total = elems
        .checked_mul(u64::from(elem))
        .filter(|&total| total <= MAX_VARIABLE_BYTES as u64)
        .ok_or_else(|| bad("PTX variable is too large"))?;
    let size = u32::try_from(total).map_err(|_| bad("PTX variable is too large"))?;
    let align = align.unwrap_or(elem).max(elem).max(4);
    if storage == ".shared" {
        if init.is_some() || size > 64 * 1024 {
            return Err(bad("unsupported PTX shared variable"));
        }
        return Ok(Declared::Shared(SharedVar {
            name,
            size: size.next_multiple_of(4),
            align: align.min(16),
        }));
    }
    if let Some(bytes) = &mut init {
        if bytes.len() as u64 > total {
            return Err(bad("PTX initializer is longer than its variable"));
        }
        bytes.resize(size as usize, 0);
        if bytes.iter().all(|&byte| byte == 0) {
            init = None;
        }
    }
    Ok(Declared::Device(GlobalVar {
        name: name.to_owned(),
        constant: storage == ".const",
        align,
        size,
        init,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<&str> {
        super::super::parse::tokens(text).unwrap()
    }

    fn device(text: &str) -> GlobalVar {
        match declaration(&tokens(text)).unwrap() {
            Declared::Device(var) => var,
            other => panic!("expected a device variable, got {other:?}"),
        }
    }

    #[test]
    fn scalar_and_zero_initialized_globals() {
        let var = device(".global .u32 counter");
        assert_eq!((var.size, var.align, var.constant), (4, 4, false));
        assert_eq!(var.init, None);
        let var = device(".visible .global .align 16 .b8 scratch[100]");
        assert_eq!((var.size, var.align), (100, 16));
        assert_eq!(var.init, None);
        let var = device(".global .u64 wide = 0");
        assert_eq!((var.size, var.align, var.init), (8, 8, None));
    }

    #[test]
    fn initializers_encode_little_endian_and_zero_fill() {
        let var = device(".const .align 4 .u32 table[4] = {1, 0x100, 3}");
        assert!(var.constant);
        assert_eq!(var.size, 16);
        assert_eq!(
            var.init.unwrap(),
            [1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
        );
        let var = device(".global .f32 pi = 0f40490FDB");
        assert_eq!(var.init.unwrap(), 0x40490fdbu32.to_le_bytes());
        let var = device(".global .f64 one = 0d3FF0000000000000");
        assert_eq!(var.init.unwrap(), 1.0f64.to_le_bytes());
        let var = device(".global .f32 half = 0.5");
        assert_eq!(var.init.unwrap(), 0.5f32.to_le_bytes());
        let var = device(".global .s16 neg[2] = {-1, 2}");
        assert_eq!(var.init.unwrap(), [0xff, 0xff, 2, 0]);
    }

    #[test]
    fn nested_braces_and_unsized_arrays() {
        let var = device(".global .u8 m[2][2] = {{1, 2}, {3}}");
        assert_eq!(var.size, 4);
        assert_eq!(var.init.unwrap(), [1, 2, 3, 0]);
        let var = device(".const .b8 blob[] = {9, 8, 7, 6, 5}");
        assert_eq!(var.size, 5);
        assert_eq!(var.init.unwrap(), [9, 8, 7, 6, 5]);
        assert!(declaration(&tokens(".global .u32 x[]")).is_err());
    }

    #[test]
    fn shared_extern_and_unsupported_forms() {
        match declaration(&tokens(".shared .align 8 .f32 tile[10]")).unwrap() {
            Declared::Shared(var) => assert_eq!((var.name, var.size, var.align), ("tile", 40, 8)),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            declaration(&tokens(".extern .shared .align 16 .b8 dyn[]")).unwrap(),
            Declared::ExternShared
        );
        assert_eq!(
            declaration(&tokens(".extern .global .b8 other[]")).unwrap(),
            Declared::Ignored
        );
        assert_eq!(
            declaration(&tokens(".global .texref tex")).unwrap(),
            Declared::Ignored
        );
        assert!(declaration(&tokens(".global .u32 x = other_symbol")).is_err());
        assert!(declaration(&tokens(".global .u8 x[2] = {1, 2, 3}")).is_err());
        assert!(declaration(&tokens(".shared .u32 s = 1")).is_err());
    }

    #[test]
    fn module_vars_keep_first_declaration() {
        let a = tokens(".global .u32 g = 5");
        let b = tokens(".global .u32 g = 6");
        let mut vars = ModuleVars::default();
        vars.record(&a);
        vars.record(&b);
        assert!(vars.contains_device("g"));
        assert_eq!(vars.device.len(), 1);
        assert_eq!(vars.device[0].init.as_deref(), Some(&5u32.to_le_bytes()[..]));
    }
}
