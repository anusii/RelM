use rand::{thread_rng, seq};
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
use rand::prelude::IteratorRandom;

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
) -> u64 {

    let weights: Vec<f64> = utilities.par_iter()
                                     .map(|u| epsilon * u / (2.0f64 * sensitivity))
                                     .map(|u| u.exp())
                                     .collect();
    let dist = WeightedIndex::new(weights).unwrap();
    samplers::discrete(&dist)
}


pub fn exponential_mechanism_gumbel_trick(
    utilities: Vec<f64>,
    sensitivity: f64,
    epsilon: f64,
) -> u64 {

    let log_weights: Vec<f64> = utilities.par_iter()
        .map(|u| epsilon * u / (2.0f64 * sensitivity))
        .collect();
    let noisy_log_weights: Vec<f64> = log_weights.par_iter()
        .map(|w| w + samplers::gumbel(1.0f64))
        .collect();
    utils::argmax(&noisy_log_weights).try_into().unwrap()
}


pub fn exponential_mechanism_sample_and_flip(
    utilities: Vec<f64>,
    sensitivity: f64,
    epsilon: f64,
) -> u64 {

    let scale: f64 = epsilon / (2.0f64 * sensitivity);
    let argmax: usize = utils::argmax(&utilities);
    let max_utility: f64 = utilities[argmax];

    let n: u64 = utilities.len().try_into().unwrap();
    let mut flag: bool = false;
    let mut current: usize = 0;
    while !flag {
        current = samplers::uniform_integer(&n).try_into().unwrap();
        let p: f64 = (scale * (utilities[current] - max_utility)).exp();
        flag = samplers::bernoulli(&p);
    }
    current.try_into().unwrap()
}


pub fn permute_and_flip_mechanism(
    utilities: Vec<f64>,
    sensitivity: f64,
    epsilon: f64,
) -> u64 {

    let scale: f64 = epsilon / (2.0f64 * sensitivity);

    let argmax: usize = utils::argmax(&utilities);
    let max_utility: f64 = utilities[argmax];

    let n: usize = utilities.len();
    let mut indices: Vec<usize> = (0..n).collect();

    indices.shuffle(&mut thread_rng());

    let bits: Vec<bool> = utilities.par_iter()
        .map(|u| (scale * (u - max_utility)).exp())
        .map(|p| samplers::bernoulli(&p))
        .collect();

    let mut shuffled_bits: Vec<bool> = vec![false; n];
    for i in 0..n {
        shuffled_bits[i] = bits[indices[i]];
    }

    let idx: usize = utils::argmax(&shuffled_bits);
    indices[idx].try_into().unwrap()

    // let mut flag: bool = false;
    // let mut idx: usize = 0;
    // let mut current: usize = 0;
    //
    // let mut rng = thread_rng();
    // while !flag {
    //     let temp = (idx..n).choose(&mut rng).unwrap();
    //     indices.swap(idx, temp);
    //     current = indices[idx];
    //     let p: f64 = (scale * (utilities[current]-max_utility)).exp();
    //     flag = samplers::bernoulli(&p);
    //     idx += 1;
    // }
    //
    // current.try_into().unwrap()
}
