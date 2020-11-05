use rand::prelude::*;
use crate::utils;


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


pub fn fixed_point_laplace(biases: &Vec<u64>, scale: f64) -> f64 {
    /// this function computes the fixed point Laplace distribution
    ///

    let mut rng = rand::thread_rng();
    let mut result: u64 = 0;

    let mut bits: u64 = rng.gen();
    let sign = 2.0 * ((bits >> 63) as f64) - 1.0;

    for idx in 0..64 {

        bits = rng.gen();

        if bits < biases[idx] {
            result |= 1 << 63 - idx;
        } else if bits > biases[idx] {
            result |= 0 << 63 - idx;
        } else {
            let pow2 = 64 - utils::PRECISION - (idx as i32) - 1;
            result |= exact_exponential_bit(scale, pow2) << 63 - idx;
        }

    }

    sign * (result as f64) * 2.0f64.powi(-utils::PRECISION)
}


fn exact_exponential_bit(scale: f64, pow2: i32) -> u64 {
    /// this function computes increasingly precise bias bits
    /// until it can be definitively determined whether the random bits
    /// are larger than the bias

    let mut rng = rand::thread_rng();
    let mut required_bits = 128;

    let mut bits: u64 = rng.gen();
    let mut bias = utils::exponential_bias(scale, pow2, required_bits);

    while bias == bits {
        required_bits += 64;
        // calculate the next 64 bits of the bias
        bias = utils::exponential_bias(scale, pow2, required_bits);
        // sample the next 64 bits from the random uniform
        bits = rng.gen();
    }

    if bias > bits {
        return 1;
    } else {
        return 0;
    }

}