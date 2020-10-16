use rand::prelude::*;


pub fn uniform(scale: f64) -> f64 {
    /// Returns a sample from the [0, scale) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}


pub fn exponential(scale: f64) -> f64 {
    /// Returns a sample from the exponential distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the exponential distribution

    let sample = -scale * uniform(1.0).ln();
    sample
}


pub fn laplace(scale: f64) -> f64 {
    /// Returns one sample from the Laplace distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the Laplace distribution

    let y = uniform(1.0) - 0.5;
    let sgn = y.signum();
    sgn * (2.0 * sgn * y).ln() * scale
}


pub fn geometric(scale: f64) -> f64 {
    /// Returns a sample from the geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the geometric distribution

    (uniform(1.0).ln() / (1.0 - scale).ln()).floor()
}


pub fn two_sided_geometric(scale: f64) -> f64 {
    /// Returns a sample from the two sided geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the two sided geometric distribution

    let y = (uniform(1.0) - 0.5) * (1.0 + scale);
    let sgn = y.signum();
    sgn * ((sgn * y).ln() / scale.ln()).floor()
}


pub fn double_uniform(scale: f64) -> f64 {
    /// Returns a sample from the [0, scale) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    let exponent: f64 = geometric(0.5) + 53.0;
    let mut significand = (rng.gen::<u64>() >> 11) | (1 << 52);
    scale * (significand as f64) * 2.0_f64.powf(-exponent)
}


pub fn fixed_point_laplace(biases: &Vec<u64>) -> f64 {
    let mut rng = rand::thread_rng();
    let mut result: u64 = 0;

    let mut bits: u64 = rng.gen();
    let mut flip_bits: u64 = 0;
    let mut offset: u32 = 0;
    let mut bit: u64 = 0;

    let mut count: u32 = 1;

    let sign = 2.0 * ((bits >> 63) as f64) - 1.0;
    bits = bits << 1;
    let start = biases.iter().position(|&x| x != 0).unwrap();

    for idx in start..64 {

        // find the first bit of disagreement between the random bits and biases
        offset = (bits ^ biases[idx]).leading_zeros();

        // if we have used up all the bits refresh and try again!
        if offset + count > 63 {
            bits = rng.gen();
            count = 0;
            offset = (bits ^ biases[idx]).leading_zeros();
        }

        // set the result idx'th bit (from left)
        // to be equal to the bias bit at the spot of disagreement
        bit = biases[idx] >> (63 - offset) & 1;
        result = result | (bit << 63 - idx);
        // keep track of the random bits consumed
        count += offset + 1;
        bits = (bits << (offset + 1));

    }
    sign * (result as f64) * 2.0f64.powi(-31)
}