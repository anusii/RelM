use rand::prelude::*;


pub fn uniform(scale: f64) -> f64 {
    /// Returns a sample from the [0, scale) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}


pub fn exponential(scale: f64) -> f64 {
    /// Returns a sample from the exponential distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the exponential distribution

    let sample = -scale * uniform(1.0).ln();
    sample
}


pub fn laplace(scale: f64) -> f64 {
    /// Returns one sample from the Laplace distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the Laplace distribution

    let y = uniform(1.0) - 0.5;
    let sgn = y.signum();
    sgn * (2.0 * sgn * y).ln() * scale
}


pub fn geometric(scale: f64) -> f64 {
    /// Returns a sample from the geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the geometric distribution

    (uniform(1.0).ln() / (1.0 - scale).ln()).floor()
}


pub fn two_sided_geometric(scale: f64) -> f64 {
    /// Returns a sample from the two sided geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the two sided geometric distribution

    let y = (uniform(1.0) - 0.5) * (1.0 + scale);
    let sgn = y.signum();
    sgn * ((sgn * y).ln() / scale.ln()).floor()
}


pub fn double_uniform(scale: f64) -> f64 {
    /// Returns a sample from the [0, scale) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    let exponent: f64 = geometric(0.5) + 53.0;
    let mut significand = (rng.gen::<u64>() >> 11) | (1 << 52);
    scale * (significand as f64) * 2.0_f64.powf(-exponent)
}


pub fn simple_coin_flip(bias: u64) -> u64 {
    let mut rng = rand::thread_rng();
    let mut bits: u64 = rng.gen();
    for ii in 0..64 {
        let shift = 63 - ii;
        if ((bits >> 63) ^ (bias >> shift)) & 1 == 1 {
            return 1 - (bits >> 63) & 1;
        }
        bits = bits << 1;
    }
    return 0;
}


pub fn coin_flip(bits: &mut u64, bias: u64, count: &mut u64) -> u64 {

    for ii in 0..64 {
        let shift = 63 - ii;
        *count += 1;
        let bit = *bits >> 63;
        if (bit ^ (bias >> shift)) & 1 == 1 {
            let result = 1 - bit;
            *bits = (*bits << 1);
            return result;
        }
        *bits = (*bits << 1);
    }
    return 0;
}



pub fn fixed_point_laplace(biases: &Vec<u64>) -> f64 {
    let mut rng = rand::thread_rng();
    let mut result: u64 = 0;

    let mut bits: u64 = rng.gen();
    let sign = 2.0 * ((bits >> 63) as f64) - 1.0;

    bits = bits << 1;
    let mut count: u64 = 1;

    for idx in 0..64 {
        if count > 50 {
            bits = rng.gen();
            count = 0;
        }

        let bit = coin_flip(&mut bits, biases[idx], &mut count);
        result = result | (bit << 63 - idx);

    }
    sign * (result as f64) * 2.0f64.powi(-31)
}