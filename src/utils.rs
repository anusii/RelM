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


pub fn fp_laplace_bit_biases(scale: f64, precision: i32) -> Vec<u64>{
    let mix_bit_bias: u64 = exponential_bias(-scale, -precision, 64);

    let mut exponential_bit_biases: [u64; 63] = [0; 63];
    for i in (0..63) {
        let pow2 = -precision + (i as i32);
        exponential_bit_biases[62-i] = exponential_bias(scale, pow2, 64);
    }

    let mut biases: Vec<u64> = Vec::new();
    biases.push(mix_bit_bias);
    biases.extend(exponential_bit_biases.iter());
    biases
}


pub fn exponential_bias(scale: f64, pow2: i32, required_bits: i32) -> u64 {
    /// this function computes the exponential bias to the specified number of
    /// required_bits and returns the least significant 64 bits.

    let num_bits = (required_bits + 10) as u32;

    let d = Float::with_val(num_bits, scale).recip() << pow2;
    let bias = (Float::with_val(num_bits, 1.0) + d.exp()).recip() << required_bits;
    bias.trunc().to_integer().unwrap().keep_bits(64).to_u64().unwrap()

}
