use rayon::prelude::*;
use crate::samplers;
use crate::utils;


pub fn all_above_threshold(data: Vec<f64>, scale: f64, threshold: f64, precision: i32) -> Vec<usize>{
    let biases: Vec<u64> = utils::fp_laplace_bit_biases(scale, precision);
    data.par_iter()
        .map(|&p| (p * 2.0f64.powi(precision)).round())
        .map(|p| (p as i64) + samplers::fixed_point_laplace(&biases, scale, precision))
        .map(|p| (p as f64) * 2.0f64.powi(-precision))
        .positions(|p| p > threshold)
        .collect()
}


pub fn snapping(data: Vec<f64>, bound: f64, lambda: f64, quanta: f64) -> Vec<f64> {
    data.par_iter()
        .map(|&p| utils::clamp(p, bound))
        .map(|p| p + lambda * utils::ln_rn(samplers::uniform_double(1.0)) * (samplers::uniform(1.0) - 0.5).signum())
        .map(|p| quanta * (p / quanta).round())
        .map(|p| utils::clamp(p, bound))
        .collect()
}


pub fn laplace_mechanism(data: Vec<f64>, sensitivity: f64, epsilon: f64, precision: i32) -> Vec<f64> {
    let scale = (sensitivity + 2.0f64.powi(-precision)) / epsilon;
    let biases: Vec<u64> = utils::fp_laplace_bit_biases(scale, precision);
    data.par_iter()
        .map(|&x| (x * 2.0f64.powi(precision)).round())
        .map(|x| (x as i64) + samplers::fixed_point_laplace(&biases, scale, precision))
        .map(|x| (x as f64) * 2.0f64.powi(-precision))
        .collect()
}


pub fn geometric_mechanism(data: Vec<i64>, sensitivity: f64, epsilon: f64) -> Vec<i64> {
    let scale = sensitivity / epsilon;
    let biases: Vec<u64> = utils::fp_laplace_bit_biases(scale, 0);
    data.par_iter()
        .map(|&x| x + samplers::fixed_point_laplace(&biases, scale, 0))
        .collect()
}


pub fn exponential_mechanism(choices: Vec<u64>, weights: Vec<f64>, k: u64) -> Vec<u64> {
    (0..k).map(|_| samplers::discrete(&choices, &weights)).collect()
}
