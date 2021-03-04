use rand::prelude::*;
use rand::distributions::{WeightedIndex, Bernoulli};
use std::convert::TryInto;

use rug::Integer;
use crate::utils;


pub fn discrete(dist: &WeightedIndex<f64>) -> u64 {
    let mut rng = rand::thread_rng();
    dist.sample(&mut rng).try_into().unwrap()
}



pub fn uniform_integer(n: u64) -> u64 {
    let mut rng = rand::thread_rng();
    let result: u64 = rng.gen_range(0, n);
    result
}

pub fn bernoulli(p: f64) -> bool {
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(p).unwrap();
    dist.sample(&mut rand::thread_rng())
}


pub fn bernoulli_log_p(log_p: f64) -> bool {
    let mut rng = rand::thread_rng();
    let mut num_required_bits = 64;

    let bias = utils::exp_rn(log_p, num_required_bits);
    let mut rand_bits = Integer::from(rng.next_u64());

    while Integer::from(&rand_bits - &bias).abs() <= 1 {
        num_required_bits += 64;
        // calculate a more precise bias
        let bias = utils::exp_rn(log_p, num_required_bits);
        // sample the next 64 bits from the random uniform
        rand_bits <<= 64;
        rand_bits += Integer::from(rng.next_u64());
    }

    bias > rand_bits
}


pub fn uniform(scale: f64) -> f64 {
    /// Samples a real from [0, scale] and rounds towards the nearest floating-point number.
    ///
    const exponent_length: u64 = 11;
    const mantissa_length: u64 = 52;
    debug_assert!(mantissa_length + exponent_length + 1 == 64);
    const exponent_mantissa_mask: u64 = (1 << (exponent_length + mantissa_length)) - 1;
    const mantissa_mask: u64 = (1 << mantissa_length) - 1;
    const max_exponent: u64 = (1 << exponent_length) - 1;
    const max_mantissa: u64 = (1 << mantissa_length) - 1;

    let scale_bits: u64 = scale.to_bits();
    let scale_exponent: u64 = (scale_bits & exponent_mantissa_mask) >> mantissa_length;
    let scale_mantissa: u64 = scale_bits & mantissa_mask;
    debug_assert!(scale_exponent <= max_exponent);
    debug_assert!(scale_mantissa <= max_mantissa);

    if scale_exponent == max_exponent || (scale_exponent == 0 && scale_mantissa == 0) {
        debug_assert!(scale.is_nan() || scale.is_infinite() || scale == 0.0);
        // As you limit x->inf, prob(sample from [0, x) > greatest float) -> 1.
        // Sub 0 to handle signalling NaNs while keeping sign.
        return scale - 0.0;
    }
    
    debug_assert!(scale != 0.0);
    debug_assert!(scale.is_finite());
    let mut rng = rand::thread_rng();

    if scale_exponent == 0 {
        debug_assert!(!scale.is_normal());
        // scale is subnormal. No need to deal with exponents since [0, scale] has
        // even intervals. Generate random mantissa in [0, scale_mantissa). Also
        // generate an extra bit for rounding direction.
        let mantissa_and_rounding: u64 = rng.gen_range(0, scale_mantissa << 1);
        let mantissa: u64 = (mantissa_and_rounding >> 1) + (mantissa_and_rounding & 1);
        let res: f64 = f64::from_bits(mantissa).copysign(scale);
        debug_assert!(res.abs() <= scale.abs());
        return res;
    }
    
    debug_assert!(scale.is_normal());
    // Scale is a normal float.
    loop { // Rejection sampling.
        // Sample from [0, 2^n) where n is the smallest integer such that scale <= 2^n.
        let rng_sample: u64 = rng.gen::<u64>();

        let rounding: u64 = rng_sample & 1;
        let mantissa: u64 = (rng_sample >> 1) & mantissa_mask;
        let mut exponent: u64 = scale_exponent - ((scale_mantissa == 0) as u64);

        // Subtract from exponent a sample from geometric distribution with p = .5
        // We still have not used the leading 11 bits of rng_sample. Re-use them to
        // avoid generating another rng sample.
        let rng_sample_geo: u64 = rng_sample & !((1 << (mantissa_length + 1)) - 1);
        if rng_sample_geo == 0 {
            exponent = exponent.saturating_sub(64 - mantissa_length - 1);
            while exponent > 0 {
                let rng_sample_inner: u64 = rng.gen::<u64>();
                if rng_sample_inner != 0 {
                    exponent = exponent.saturating_sub(rng_sample_inner.leading_zeros() as u64);
                    break;
                }
                exponent = exponent.saturating_sub(64);
            }
        } else {
            exponent = exponent.saturating_sub(rng_sample_geo.leading_zeros() as u64);
        }

        debug_assert!(exponent <= scale_exponent);
        if exponent < scale_exponent || mantissa < scale_mantissa {
            let res: f64 = f64::from_bits((exponent << mantissa_length)
                                          + mantissa
                                          + rounding).copysign(scale);
            debug_assert!(res.abs() <= scale.abs());
            return res;
        }

        debug_assert!(f64::from_bits((exponent << mantissa_length) + mantissa + rounding)
                      > scale.abs());
        // result > scale; rejecting.
    }
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
