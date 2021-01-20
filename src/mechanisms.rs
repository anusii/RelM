use rand::{thread_rng, Rng};
use rand::distributions::WeightedIndex;
use rand::seq::IteratorRandom;

use std::convert::TryInto;
use std::collections::HashMap;

use rayon::prelude::*;
use crate::samplers;
use crate::utils;


pub fn all_above_threshold(data: Vec<f64>, epsilon: f64, threshold: f64, precision: i32) -> Vec<usize>{
    let scale = 1.0 / epsilon;
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


pub fn laplace_mechanism(data: Vec<f64>, epsilon: f64, precision: i32) -> Vec<f64> {
    let scale = 1.0 / epsilon;
    let biases: Vec<u64> = utils::fp_laplace_bit_biases(scale, precision);
    data.par_iter()
        .map(|&x| (x * 2.0f64.powi(precision)).round())
        .map(|x| (x as i64) + samplers::fixed_point_laplace(&biases, scale, precision))
        .map(|x| (x as f64) * 2.0f64.powi(-precision))
        .collect()
}


pub fn geometric_mechanism(data: Vec<i64>, epsilon: f64) -> Vec<i64> {
    let scale = 1.0 / epsilon;
    let biases: Vec<u64> = utils::fp_laplace_bit_biases(scale, 0);
    data.par_iter()
        .map(|&x| x + samplers::fixed_point_laplace(&biases, scale, 0))
        .collect()
}


pub fn exponential_mechanism_weighted_index(
    utilities: Vec<f64>,
    epsilon: f64,
) -> u64 {

    let weights: Vec<f64> = utilities.par_iter()
                                     .map(|u| epsilon * u)
                                     .map(|u| u.exp())
                                     .collect();
    let dist = WeightedIndex::new(weights).unwrap();
    samplers::discrete(&dist)
}


pub fn exponential_mechanism_gumbel_trick(
    utilities: Vec<f64>,
    epsilon: f64,
) -> u64 {

    let log_weights: Vec<f64> = utilities.par_iter()
        .map(|u| epsilon * u)
        .collect();
    let noisy_log_weights: Vec<f64> = log_weights.par_iter()
        .map(|w| w + samplers::gumbel(1.0f64))
        .collect();
    utils::argmax(&noisy_log_weights).try_into().unwrap()
}


pub fn exponential_mechanism_sample_and_flip(
    utilities: Vec<f64>,
    epsilon: f64,
) -> u64 {

    let scale: f64 = epsilon;
    let argmax: usize = utils::argmax(&utilities);
    let max_utility: f64 = utilities[argmax];

    let n: u64 = utilities.len().try_into().unwrap();
    let mut flag: bool = false;
    let mut current: usize = 0;
    while !flag {
        current = samplers::uniform_integer(n).try_into().unwrap();
        let log_p = scale * (utilities[current] - max_utility);
        flag = samplers::bernoulli_log_p(log_p);
    }
    current.try_into().unwrap()
}


pub fn permute_and_flip_mechanism(
    utilities: Vec<f64>,
    epsilon: f64,
) -> u64 {

    let scale: f64 = epsilon;

    let argmax: usize = utils::argmax(&utilities);
    let max_utility: f64 = utilities[argmax];

    let mut normalized_log_weights: Vec<f64> = utilities.par_iter()
        .map(|u| (scale * (u - max_utility)))
        .collect();

    let n: usize = utilities.len();
    let mut indices: Vec<usize> = (0..n).collect();

    let mut rng = thread_rng();
    let mut flag: bool = false;
    let mut idx: usize = 0;
    let mut current: usize = 0;
    while !flag {
        let temp = rng.gen_range(idx, n);
        indices.swap(idx, temp);
        current = indices[idx];
        flag = samplers::bernoulli_log_p(normalized_log_weights[current]);
        idx += 1;
    }
    current.try_into().unwrap()
}


pub fn small_db(
    epsilon: f64, l1_norm: usize, size: u64, queries: Vec<u64>, answers: Vec<f64>, breaks: Vec<usize>
) -> Vec<u64> {

    // store the db in a sparse vector (implemented with a HashMap)
    let mut db: HashMap<u64, u64> = HashMap::with_capacity(l1_norm);

    // let mut normalizer: f64 = f64::MIN;
    // for i in 0..1000 {
    //     random_small_db(&mut db, l1_norm, size);
    //
    //     let error = small_db_max_error(&db, &queries, &answers, &breaks, l1_norm);
    //     let utility = -error * (l1_norm as f64);
    //     println!("{:?}", utility);
    //     println!("{:?}", normalizer);
    //
    //     if utility > normalizer {
    //         normalizer = utility;
    //     }
    // }

    // let unnormalized_answers = answers.iter().map(|x| x * (l1_norm as f64));

    let min_errors = answers.iter()
                            .map(|x| x * (l1_norm as f64))
                            .map(|x| (x - x.round()).abs());
    let mut max_min_error: f64 = 0.0;
    for (i, min_error) in min_errors.enumerate() {
        if min_error > max_min_error {
            max_min_error = min_error;
        }
    }

    let mut normalizer: f64 = -max_min_error;

    loop {
        // sample another random small db
        random_small_db(&mut db, l1_norm, size);

        let error = small_db_max_error(&db, &queries, &answers, &breaks, l1_norm);
        let utility = -error * (l1_norm as f64);
        //println!("{:?}", utility);
        //println!("{:?}", normalizer);

        let log_p = epsilon * (utility - normalizer);
        if samplers::bernoulli_log_p(log_p) { break }

        // if utility > normalizer {
        //     normalizer = utility;
        // } else {
        //     let log_p = epsilon * (utility - normalizer);
        //     if samplers::bernoulli_log_p(log_p) { break }
        // }
    }

    // convert the sparse small db to a dense vector
    let mut db_vec: Vec<u64> = vec![0; size as usize];
    for (&idx, &val) in db.iter() {
        db_vec[idx as usize] = val;
    }

    db_vec
}


fn random_small_db(db: &mut HashMap<u64, u64>, l1_norm: usize, size: u64) {
    /// generates a random sparse database with size `size` and a norm of `l1_norm` in place
    /// overwrites previous db for time and space efficiency

    db.clear();
    let mut rng = thread_rng();

    // Use the "stars and bars" approach to generate a random database
    let num_bars: usize = (size as usize) - 1;
    let slots: usize = l1_norm + num_bars;
    let mut partitions: Vec<i64> = vec![0; num_bars + 2];
    partitions[0]= -1;
    let mut temp = (0..slots).choose_multiple(&mut rng, num_bars);
    temp.sort();
    for (i,idx) in temp.iter().enumerate() {
        partitions[i+1] = *idx as i64;
    }
    partitions[num_bars+1] = slots as i64;

    let mut values: Vec<u64> = vec![0; size as usize];
    for i in 0..size.try_into().unwrap() {
        values[i] = ((partitions[i+1] - partitions[i]) - 1) as u64;
    }

    for i in 0..size.try_into().unwrap() {
        let value: u64 = ((partitions[i+1] - partitions[i]) - 1) as u64;
        if value > 0 {
            let key: u64 = i as u64;
            db.insert(key, value);
        }
    }
}


fn small_db_max_error(
    db: &HashMap<u64, u64>, queries: &Vec<u64>, answers: &Vec<f64>, breaks: &Vec<usize>, l1_norm: usize
) -> f64 {

    let mut max_error: f64 = 0.0;
    let mut result: u64 = 0;
    let mut error: f64 = 0.0;

    let mut start: usize = 0;

    // breaks determines the index of `queries` at which the distinct queries end/start
    // iterate through queries
    for (i, &stop) in breaks.iter().enumerate() {
        // calculate result of query
        result = 0;
        // iterate through the indices stored in the query
        for j in start..stop {
            let idx = queries[j];
            result += match db.get(&idx) {
                Some(x) => {*x}
                None => 0
            };
        }

        start = stop;

        // store largest error
        let normalized_result = (result as f64) / (l1_norm as f64);
        error = (normalized_result - answers[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    max_error
}
