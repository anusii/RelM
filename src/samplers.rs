use rand::prelude::*;
// use rand::distributions::{WeightedIndex, Bernoulli};
use rand_distr::{WeightedIndex, Bernoulli, Cauchy, Distribution};
use std::convert::TryInto;

use rug::Integer;
use crate::utils;


pub fn discrete(dist: &WeightedIndex<f64>) -> u64 {
    let mut rng = rand::thread_rng();
    dist.sample(&mut rng).try_into().unwrap()
}


pub fn uniform_integer(n: u64) -> u64 {
    let mut rng = rand::thread_rng();
    let result: u64 = rng.gen_range(0..n);
    result
}


pub fn bernoulli(p: f64) -> bool {
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(p).unwrap();
    dist.sample(&mut rng)
}


pub fn cauchy(scale: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let dist = Cauchy::new(0.0, scale).unwrap();
    dist.sample(&mut rng)
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


fn capped_geometric2(cap: u64, rng: &mut rand::rngs::ThreadRng) -> u64 {
    /// Samples an integer from the geometric distribution with success
    /// probability = .5. Results greater than `cap` saturate to `cap`.
    ///
    let mut res = 0;
    while res < cap {
        let sample = rng.next_u64();
        if sample != 0 {
            res += sample.trailing_zeros() as u64;
            break;
        }
        res = res.saturating_add(64);
    }
    if res >= cap {
        cap
    } else {
        res
    }
}



const F64_EXPONENT_LEN: u64 = 11;
const F64_MANTISSA_LEN: u64 = 52;
const F64_MAX_EXPONENT: u64 = (1 << F64_EXPONENT_LEN) - 1;
const F64_MAX_MANTISSA: u64 = (1 << F64_MANTISSA_LEN) - 1;


fn extract_bits(x: u64, i: u64, len: u64) -> u64 {
    // Returns len bits from x, beginning at index i.
    // The least-significant bit has index 0.
    (x >> i) & ((1 << len) - 1)
}

fn decompose_float(x: u64) -> (u64, u64, u64) {
    let sign = x >> F64_EXPONENT_LEN + F64_MANTISSA_LEN;
    let exponent = x >> F64_MANTISSA_LEN & F64_MAX_EXPONENT;
    let mantissa = x & F64_MAX_MANTISSA;
    (sign, exponent, mantissa)
}


pub fn uniform(scale: f64) -> f64 {
    /// Samples a real from [0, scale] and rounds towards the nearest floating-point number.
    ///
    if scale == 0.0 {
        return scale; // Return zero of the same sign (:
    }
    if scale.is_infinite() {
        return scale; // As scale->inf, p(sample > greatest float)->1.
    }
    if scale.is_nan() {
        return scale + 0.0; // +0 to silence signalling NaN.
    }

    let (_, scale_exponent, scale_mantissa) = decompose_float(scale.to_bits());
    let mut rng = rand::thread_rng();

    if scale_exponent == 0 {
        // Scale is subnormal.
        // Let s be the smallest nonzero subnormal.
        // The floats between 0 and scale lie at 0, s, 2 * s, ..., n * s = scale for some n.
        // We wish to sample a real in (0, scale) and round to the nearest float.
        // A sample in (0, s/2) gets rounded to 0, and a sample in (n * s - s/2, n * s) rounds to
        // scale. Finally, for all i = 1, ..., n - 1, (i * s - s/2, i * s + s/2) rounds to i * s.
        // Note that we don't need to worry about the case of a real sample being exactly between
        // two floats, because that event has probability 0.
        // We do this in two steps:
        //  1. Choose an interval between two floating point numbers. The choices are:
        //     (0, s), (s, 2 * s), ..., ((n - 1) * s, n * s). Each of these intervals has equal size
        //     so we can sample uniformly in {0, ..., n - 1}. This interval is represented by the
        //     `mantissa` below.
        //  2. We now have some integer i (`mantissa`) such that we need to sample from
        //     (i * s, (i + 1) * s) and round to the nearest float. This interval is exactly the
        //     space between two floating point numbers. Since we're rounding to nearest, we'll
        //     round to i * s and to (i + 1) * s with equal probability. We can sample uniformly
        //     j in {0, 1} (`rounding` below) and return (i + j) * s.
        debug_assert!(!scale.is_normal());
        let scale_mantissa_x2 = scale_mantissa << 1;
        let mantissa_and_rounding: u64 = rng.gen_range(0..scale_mantissa_x2);
        let mantissa: u64 = (mantissa_and_rounding >> 1) + (mantissa_and_rounding & 1);
        let res: f64 = f64::from_bits(mantissa).copysign(scale);
        debug_assert!(res.abs() <= scale.abs());
        return res;
    }

    debug_assert!(scale.is_normal());
    // Scale is a normal float.
    // Let b be the smallest power of 2 such that scale <= b. We wish to sample a real from
    // (0, scale) and round to the nearest float. We achieve this by sampling from (0, b), rejecting
    // if the sample > scale, and rounding an accepted sample (in (0, scale)) to the nearest float.
    // Observe that b is no smaller than the smallest normal float (but it can be greater than the
    // greatest normal float).
    //
    // Let c be a power of 2. To sample from (0, c), we work by cases.
    //
    // If c is the smallest normal float, then we follow similar logic to the subnormal case above.
    // The possible results are 0, s, 2 * s, ..., n * s = c. We first select an interval between two
    // floating-point numbers by sampling uniformly i from {0, ..., n - 1}. We've thus narrowed down
    // the problem to sampling from (i * s, (i + 1) * s) and rounding to the nearest float. Since
    // the bottom half of that interval rounds down and the top half rounds up, we sample uniformly
    // j in {0, 1} and return (i + j) * s.
    //
    // Otherwise, c is a power of 2 that is not the smallest normal float. We want to sample from
    // (0, c). We do this by first drawing a Bernoulli sample. On success we recurse and sample from
    // (0, c/2). Observe that c/2 is also a power of 2 no smaller than the smallest normal float.
    // This is equivalent to decreasing the exponent of c by 1. On failure, we sample from (c/2, c)
    // and round to the nearest float.
    //
    // Observe that the floating-point numbers in are evently distributed with a step of s. They
    // are c/2, c/2 + s, c/2 + 2 * s, ..., c/2 + 2^n = c. (As an exception, sometimes c may not be
    // representable as a floating point number. This has no impact on the argument.) Each of these
    // floats has the same exponent, except c which has exponent 1 bigger. We first choose an
    // interval between two floats by sampling i uniformly in {0, ..., 2^n - 1}. We now know that
    // our real sample is in some interval (c/2 + i * s, c/2 + (i + 1) * s). If
    // c/2 + i * s >= scale, then our real sample > scale, and we reject. Otherwise we round to
    // either c/2 + i * s or c/2 + (i + 1) * s with equal probability.

    loop { // Rejection sampling.
        // Sample from [0, 2^n) where n is the smallest integer such that scale <= 2^n.
        let rng_sample: u64 = rng.next_u64();
        let (rounding, rng_sample_geo, mantissa) = decompose_float(rng_sample);
        // scale_mantissa == 0 means scale is a power of 2. In that case the exponent of the sample
        // < scale_exponent unless sample == scale (this only happens if mantissa is all ones and we
        // round up; it is handled by addition with carry).
        let mut exponent: u64 = scale_exponent - ((scale_mantissa == 0) as u64);

        // Subtract from exponent a sample from geometric distribution with p = .5
        // We still have not used the leading 11 bits of rng_sample. Re-use them to
        // avoid generating another rng sample.
        if rng_sample_geo == 0 {
            // Saturating at exponent = 0 is equivalent to no longer splitting intervals when we
            // reach subnormal numbers.
            exponent = exponent.saturating_sub(F64_EXPONENT_LEN);
            // All our samples so far have been successes. Keep drawing until failure or until the
            // exponent reaches 0.
            exponent -= capped_geometric2(exponent, &mut rng);
        } else {
            exponent = exponent.saturating_sub(rng_sample_geo.trailing_zeros() as u64);
        }

        debug_assert!(exponent <= scale_exponent);
        if exponent < scale_exponent || mantissa < scale_mantissa {
            // result < scale; accept
            // Important to carry overflow from the mantissa to the exponent.
            let res: f64 = f64::from_bits((exponent << F64_MANTISSA_LEN)
                                          + mantissa
                                          + rounding).copysign(scale);
            debug_assert!(res.abs() <= scale.abs());
            return res;
        }

        debug_assert!(f64::from_bits((exponent << F64_MANTISSA_LEN) + mantissa + rounding)
                      >= scale.abs());
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
