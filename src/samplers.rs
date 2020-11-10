use rand::prelude::*;
use rug::Integer;
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


pub fn fixed_point_laplace(biases: &Vec<u64>, scale: f64, precision: i32) -> i64 {
    /// this function computes the fixed point Laplace distribution
    ///

    let mut exponential_bits: i64 = 0;
    let mut pow2: i32 = 0;

    let mix_bit = sample_exponential_bit(biases[0], -scale, -precision);

    for idx in 1..64 {
        pow2 = 64 - precision - (idx as i32) - 1;
        let temp = sample_exponential_bit(biases[idx], scale, pow2);
        exponential_bits |= temp << (63 - idx);
    }

    let laplace_bits = (-1 + mix_bit) ^ exponential_bits;
    laplace_bits
}


fn sample_exponential_bit(bias: u64, scale: f64, pow2: i32) -> i64 {
    let mut rng = rand::thread_rng();
    let mut exponential_bit: i64 = 0;
    let rand_bits: u64 = rng.gen();

    if rand_bits.saturating_sub(bias) + bias.saturating_sub(rand_bits) <= 1 {
        exponential_bit = sample_exact_exponential_bit(scale, pow2, rand_bits);
    } else if rand_bits < bias {
        exponential_bit = 1;
    } else if rand_bits > bias {
        exponential_bit = 0;
    } else {
        panic!("Error: code should never reach here.");
    }

    exponential_bit
}


fn sample_exact_exponential_bit(scale: f64, pow2: i32, rand_bits: u64) -> i64 {
    /// this function computes increasingly precise bias bits
    /// until it can be definitively determined whether the random bits
    /// are larger than the bias

    let mut rng = rand::thread_rng();
    let mut num_required_bits = 128;

    let bias = utils::exponential_bias(scale, pow2, num_required_bits);

    let mut rand_bits = Integer::from(rand_bits) << 64;
    rand_bits += Integer::from(rng.next_u64());

    while Integer::from((&rand_bits - &bias)).abs() <= 1 {
        num_required_bits += 64;
        // calculate a more precise bias
        let bias = utils::exponential_bias(scale, pow2, num_required_bits);
        // sample the next 64 bits from the random uniform
        rand_bits <<= 64;
        rand_bits += Integer::from(rng.next_u64());
    }

    if bias > rand_bits {
        return 1;
    } else {
        return 0;
    }
}
