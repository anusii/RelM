use rug::{float::Round, Float};
use rayon::prelude::*;


pub fn clamp(x: f64, bound: f64) -> f64 {
    if x < -bound {
        -bound
    } else if x > bound {
        bound
    } else {
        x
    }
}


pub fn ln_rn(x: f64) -> f64 {
    let mut y = Float::with_val(53, x);
    let dir = y.ln_round(Round::Nearest);
    y.to_f64()
}


pub fn vectorize(scale: f64, num: usize, func: fn(f64) -> f64) -> Vec<f64> {
    /// Vectorize a distribution sampler
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the distribution
    /// * `num` - The number of samples to draw
    /// * `func` - The distribution function

    let mut samples: Vec<f64> = vec![0.0; num];
    samples.par_iter_mut().for_each(|p| *p = func(scale));
    samples
}


pub fn exponential_biases(scale: f64) -> Vec<u64>{
    (0..64).map(|i| exponential_bias(scale, 32 - i, 64)).collect()
}


pub fn exponential_bias(scale: f64, pow2: i32, precision: i32) -> u64 {
    /// this function computes the exponential bias to the specified precision and
    /// returns the least significant 64 bits

    let num_bits = (precision + 10) as u32;

    let d = Float::with_val(num_bits, 2.0f64.powi(pow2)) / Float::with_val(num_bits, scale);

    let mut bias = Float::with_val(num_bits, 1.0) / (Float::with_val(num_bits, 1.0) + d.exp());
    bias *= Float::with_val(num_bits, 2.0f64.powi(precision));

    // remove the most significant (precision - 64) bits
    let mut bias = bias.to_integer().unwrap();
    bias.keep_bits_mut(64);
    bias.to_u64().unwrap()
}