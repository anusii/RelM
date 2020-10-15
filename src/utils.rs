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