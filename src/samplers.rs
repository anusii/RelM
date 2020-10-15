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