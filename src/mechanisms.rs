use rayon::prelude::*;
use crate::samplers;
use crate::utils;


pub fn all_above_threshold(
    data: Vec<f64>, scale: f64, threshold: f64
) -> Vec<usize>{
    data.par_iter().positions(|&p| p + samplers::laplace(scale) > threshold).collect()
}


pub fn snapping(
    data: Vec<f64>, bound: f64, lambda: f64, quanta: f64
) -> Vec<f64> {
    data.par_iter()
        .map(|&p| utils::clamp(p, bound))
        .map(|p| p + lambda * utils::ln_rn(samplers::double_uniform(1.0)) * (samplers::uniform(1.0) - 0.5).signum())
        .map(|p| quanta * (p / quanta).round())
        .map(|p| utils::clamp(p, bound))
        .collect()
}
