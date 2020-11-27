use rand::distributions::WeightedIndex;
use std::convert::TryInto;

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


pub fn exponential_mechanism_weighted_index(
    utilities: Vec<f64>,
    sensitivity: f64,
    epsilon: f64,
    k: usize)
-> Vec<u64> {

    let weights: Vec<f64> = utilities.par_iter()
                                     .map(|u| epsilon * u / (2.0f64 * sensitivity))
                                     .map(|u| u.exp())
                                     .collect();
    let dist = WeightedIndex::new(weights).unwrap();
    let n: u64 = utilities.len().try_into().unwrap();
    let choices: Vec<u64> = (0..n).collect();
    (0..k).map(|_| samplers::discrete(&choices, &dist)).collect()
}


pub fn exponential_mechanism_gumbel_trick(
    utilities: Vec<f64>,
    sensitivity: f64,
    epsilon: f64,
    k: usize)
-> Vec<u64> {

    let log_weights: Vec<f64> = utilities.par_iter()
        .map(|u| epsilon * u / (2.0f64 * sensitivity))
        .collect();
    let mut indices: Vec<u64> = vec![0; k];
    for i in 0..k {
        let noisy_log_weights: Vec<f64> = log_weights.par_iter()
            .map(|w| w + samplers::gumbel(1.0f64))
            .collect();
        indices[i] = utils::argmax(&noisy_log_weights).try_into().unwrap();
    }
    indices
}


pub fn exponential_mechanism_sample_and_flip(
    utilities: Vec<f64>,
    sensitivity: f64,
    epsilon: f64,
    k: usize)
-> Vec<u64> {

    let log_weights: Vec<f64> = utilities.par_iter()
        .map(|u| epsilon * u / (2.0f64 * sensitivity))
        .collect();
    let n: u64 = utilities.len().try_into().unwrap();
    let mut indices: Vec<u64> = vec![0; k];
    for i in 0..k {
        let mut flag: bool = false;
        while !flag {
            let index: u64 = samplers::uniform_integer(&n);
            let p: f64 = (epsilon * log_weights[i] / (2.0f64 * sensitivity)).exp();
            flag = samplers::bernoulli(&p);
            if flag {
                indices[i] = index;
            }
        }
    }
    indices
}
