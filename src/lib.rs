use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rand::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray, PyArray1, ToPyArray};


fn uniform() -> f64 {
    /// Returns a sample from the [0, 1) uniform distribution
    ///

    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}


fn exponential(scale: f64) -> f64 {
    /// Returns a sample from the exponential distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the exponential distribution

    let sample = -scale * uniform().ln();
    sample
}


fn laplace(scale: f64) -> f64 {
    /// Returns one sample from the Laplace distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the Laplace distribution

    let y = uniform() - 0.5;
    let sgn = y.signum();
    sgn * (2.0 * sgn * y).ln() * scale
}


fn geometric(scale: f64) -> f64 {
    /// Returns a sample from the geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the geometric distribution

    (uniform().ln() / (1.0 - scale).ln()).floor()
}


fn two_sided_geometric(scale: f64) -> f64 {
    /// Returns a sample from the two sided geometric distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the two sided geometric distribution

    let y = (uniform() - 0.5) * (1.0 + scale);
    let sgn = y.signum();
    sgn * ((sgn * y).ln() / scale.ln()).floor()
}


fn double_uniform() -> f64 {
    /// Returns a sample from the [0, 1) uniform distribution
    ///
    
    let mut rng = rand::thread_rng();
    let exponent: f64 = geometric(0.5) + 53.0;
    let mut significand = (rng.gen::<u64>() >> 11) | (2 << 51);
    (significand as f64) * 2.0_f64.powf(-exponent)
}


fn vectorize(scale: f64, num: usize, func: fn(f64) -> f64) -> Vec<f64> {
    /// Vectorize a distribution sampler
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the distribution
    /// * `num` - The number of samples to draw
    /// * `func` - The distribution function

    let mut samples: Vec<f64> = vec![0.0; num];
    samples.par_iter_mut().for_each(|p| *p = func(scale));
    samples
}


fn all_above_threshold(
    data: Vec<f64>, scale: f64, threshold: f64
) -> Vec<usize>{
    data.par_iter().positions(|&p| p + laplace(scale) > threshold).collect()
}


///// A Python module implemented in Rust.
///// Exports the rust functions to python.
#[pymodule]
fn backend(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "uniform")]
    fn py_uniform(py: Python, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let mut samples: Vec<f64> = vec![0.0; num];
        samples.par_iter_mut().for_each(|p| *p = uniform());
        samples.to_pyarray(py)
    }

    #[pyfn(m, "exponential")]
    fn py_exponential(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array

        vectorize(scale, num, exponential).to_pyarray(py)
    }

    #[pyfn(m, "laplace")]
    fn py_laplace(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the laplace function. Converts
        /// the rust vector into a numpy array

        vectorize(scale, num, laplace).to_pyarray(py)
    }

    #[pyfn(m, "geometric")]
    fn py_geometric(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the geometric function. Converts
        /// the rust vector into a numpy array

        vectorize(scale, num, geometric).to_pyarray(py)
    }

    #[pyfn(m, "two_sided_geometric")]
    fn py_two_sided_geometric(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the two sided geometric function. Converts
        /// the rust vector into a numpy array

        vectorize(scale, num, two_sided_geometric).to_pyarray(py)
    }

    #[pyfn(m, "double_uniform")]
    fn py_double_uniform(py: Python, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let mut samples: Vec<f64> = vec![0.0; num];
        samples.par_iter_mut().for_each(|p| *p = double_uniform());
        samples.to_pyarray(py)
    }

    #[pyfn(m, "all_above_threshold")]
    fn py_all_above_threshold<'a>(
        py: Python<'a>, data: &'a PyArray1<f64>,
        scale: f64, threshold: f64
    ) -> &'a PyArray1<usize> {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let data = data.to_vec().unwrap();
        all_above_threshold(data, scale, threshold).to_pyarray(py)
    }

    Ok(())

}