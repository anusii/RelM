use rand::prelude::*;
use rand::distributions::{WeightedIndex, Bernoulli};

use rug::Integer;
use crate::utils;


pub fn discrete(choices: &Vec<u64>, dist: &WeightedIndex<f64>) -> u64 {
    let mut rng = rand::thread_rng();
    choices[dist.sample(&mut rng)]
}


pub fn uniform_integer(n: &u64) -> u64 {
    let mut rng = rand::thread_rng();
    let result: u64 = rng.gen_range(0, *n);
    result
}

pub fn bernoulli(p: &f64) -> bool {
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(*p).unwrap();
    dist.sample(&mut rand::thread_rng())
}

pub fn uniform(scale: f64) -> f64 {
    /// Returns a sample from the [0, scale) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    scale * rng.gen::<f64>()
}


pub fn geometric(scale: f64) -> f64 {
    /// Returns a sample from the geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the geometric distribution
    let mut rng = rand::thread_rng();
    (rng.gen::<f64>().ln() / (1.0 - scale).ln()).floor()
}


pub fn gumbel(scale: f64) -> f64 {
    /// Returns a sample from the Gumbel distribution
    ///
    /// # Arguments
    ///
    /// * `scale` = The scale parameter of the Gumbel distribution
    let mut rng = rand::thread_rng();
    -scale * (-rng.gen::<f64>().ln()).ln()
}


pub fn uniform_double(scale: f64) -> f64 {
    /// Returns a sample from the [0, scale) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    let exponent: f64 = geometric(0.5) + 53.0;
    let significand = (rng.gen::<u64>() >> 11) | (1 << 52);
    scale * (significand as f64) * 2.0_f64.powf(-exponent)
}


pub fn fixed_point_exponential(biases: &Vec<u64>, scale: f64, precision: i32) -> i64 {
    /// this function computes the fixed point exponential distribution
    ///

    let mut rng = thread_rng();

    let mut exponential_bits: i64 = 0;
    let mut pow2: i32;

    for idx in 1..64 {
        let rand_bits = rng.next_u64();
        pow2 = 64 - precision - (idx as i32) - 1;
        let bit = match comp_exp_bit(biases[idx], rand_bits) {
            Some(x) => x,
            None => sample_exact_exponential_bit(scale, pow2, rand_bits)
        };
        exponential_bits |= bit << (63 - idx);
    }

    exponential_bits
}

pub fn fixed_point_laplace(biases: &Vec<u64>, scale: f64, precision: i32) -> i64 {
    /// this function computes the fixed point Laplace distribution
    ///

    let mut rng = thread_rng();

    let rand_bits = rng.next_u64();
    let mix_bit = match comp_exp_bit(biases[0], rand_bits) {
        Some(x) => x,
        None => sample_exact_exponential_bit(-scale, -precision, rand_bits)
    };

    let exponential_bits = fixed_point_exponential(&biases, scale, precision);

    let laplace_bits = (-1 + mix_bit) ^ exponential_bits;
    laplace_bits
}


fn comp_exp_bit(bias: u64, rand_bits: u64) -> Option<i64> {
    if rand_bits.saturating_sub(bias) + bias.saturating_sub(rand_bits) <= 1 {
        None
    } else if rand_bits < bias {
        Some(1)
    } else if rand_bits > bias {
        Some(0)
    } else {
        panic!("Error: code should never reach here.");
    }
}


fn sample_exact_exponential_bit(scale: f64, pow2: i32, rand_bits: u64) -> i64 {
    /// this function computes increasingly precise bias bits
    /// until it can be definitively determined whether the random bits
    /// are larger than the bias

    let mut rng = rand::thread_rng();
    let mut num_required_bits = 128;

    let mut bias = utils::exponential_bias(scale, pow2, num_required_bits);

    let mut rand_bits = Integer::from(rand_bits) << 64;
    rand_bits += Integer::from(rng.next_u64());

    while Integer::from(&rand_bits - &bias).abs() <= 1 {
        num_required_bits += 64;
        // calculate a more precise bias
        bias = utils::exponential_bias(scale, pow2, num_required_bits);
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


#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_sample_exact_exponential_bit() {
        let scale: f64 = 1.0;
        let pow2 = 1;
        let mut rng = thread_rng();
        let rand_bits: u64 = utils::exponential_bias(scale, pow2, 64).to_u64().unwrap();

        sample_exact_exponential_bit(scale, pow2, rand_bits);
    }
}
