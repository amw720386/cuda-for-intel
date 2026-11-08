//! Small non-cryptographic 128-bit content hash (two FNV-1a lanes) for cache keys.

pub struct Hasher {
    a: u64,
    b: u64,
}

impl Default for Hasher {
    fn default() -> Self {
        Self { a: 0xcbf2_9ce4_8422_2325, b: 0x8422_2325_cbf2_9ce4 ^ 0x9e37_79b9_7f4a_7c15 }
    }
}

impl Hasher {
    pub fn bytes(&mut self, data: &[u8]) -> &mut Self {
        // Length-prefix every field so ("ab","c") and ("a","bc") differ.
        self.raw(&(data.len() as u64).to_le_bytes());
        self.raw(data);
        self
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.bytes(s.as_bytes())
    }
    fn raw(&mut self, data: &[u8]) {
        for &x in data {
            self.a = (self.a ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3);
            self.b = (self.b ^ u64::from(x).rotate_left(3)).wrapping_mul(0x0000_0100_0000_01b3 ^ 0x2545_f491);
        }
    }
    pub fn finish(&self) -> String {
        format!("{:016x}{:016x}", self.a, self.b)
    }
}
