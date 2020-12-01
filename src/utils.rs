use rug::{float::Round, Float, Integer};

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
    y.ln_round(Round::Nearest);
    y.to_f64()
}


pub fn argmax<T: std::cmp::PartialOrd + Copy>(slice: &Vec<T>) -> usize {
    let mut max_val = slice[0];
    let mut max_idx: usize = 0;
    let mut idx = 0;
    for &val in slice {
        if val > max_val {
            max_idx = idx;
            max_val = val;
        }
        idx += 1;
    }

    max_idx
}


pub fn fp_laplace_bit_biases(scale: f64, precision: i32) -> Vec<u64>{

    let mut biases: Vec<u64> = vec![0; 64];
    let mix_bit_bias = exponential_bias(-scale, -precision, 64).to_u64().unwrap();
    biases[0] = mix_bit_bias;

    let mut pow2 = 62 - precision;
    for i in 1..64 {
        biases[i] = exponential_bias(scale, pow2, 64).to_u64().unwrap();
        pow2 -= 1;
    }

    biases
}


pub fn exponential_bias(scale: f64, pow2: i32, required_bits: i32) -> Integer {
    /// this function computes the exponential bias to the specified number of
    /// required_bits

    let num_bits = (required_bits + 10) as u32;

    let d = Float::with_val(num_bits, scale).recip() << pow2;
    let bias = (Float::with_val(num_bits, 1.0) + d.exp()).recip() << required_bits;
    bias.trunc().to_integer().unwrap()
}
