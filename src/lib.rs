use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rand::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray, PyArray1, ToPyArray};


fn uniform() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}


fn exponential(scale: f64, num: usize) -> Vec<f64> {
    /// Returns num samples from the exponential distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the exponential distribution
    /// * `num` - The number of samples to draw from the exponential distribution
    let mut samples: Vec<f64> = vec![0.0; num];
    samples.par_iter_mut().for_each(|p| *p = -scale * uniform().ln());
    samples
}


fn scalar_laplace(scale: f64) -> f64 {
    /// Returns one sample from the Laplace distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the Laplace distribution
    let y = uniform() - 0.5;
    let sgn = y.signum();
    sgn * (2.0 * sgn * y).ln() * scale
}


fn laplace(scale: f64, num: usize) -> Vec<f64> {
    /// Returns num samples from the Laplace distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the Laplace distribution
    /// * `num` - The number of samples to draw from the Laplace distribution
    let mut samples: Vec<f64> = vec![0.0; num];
    samples.par_iter_mut().for_each(|p| *p = scalar_laplace(scale));
    samples
}


fn geometric(param: f64, num: usize) -> Vec<f64> {
    /// Returns num samples from the geometric distribution
    ///
    /// # Arguments
    ///
    /// * `p` - The parameter of the geometric distribution
    /// * `num` - The number of samples to draw from the geometric distribution
    let mut samples: Vec<f64> = vec![0.0; num];
    let c = (1.0 - param).ln();
    samples.par_iter_mut().for_each(|p| *p = (uniform().ln() / c).floor());
    samples
}


fn scalar_two_sided_geometric(param: f64) -> f64 {
    let y = (uniform() - 0.5) * (1.0 + param);
    let sgn = y.signum();
    sgn * ((sgn * y).ln() / param.ln()).floor()
}


fn two_sided_geometric(param: f64, num: usize) -> Vec<f64> {
    /// Returns num samples from the two sided geometric distribution
    ///
    /// # Arguments
    ///
    /// * `p` - The parameter of the geometric distribution
    /// * `num` - The number of samples to draw from the geometric distribution
    let mut samples: Vec<f64> = vec![0.0; num];
    samples.par_iter_mut().for_each(|p| *p = scalar_two_sided_geometric(param));
    samples
}

///// A Python module implemented in Rust.
///// Exports the rust functions to python.
#[pymodule]
fn differential_privacy(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "exponential")]
    fn py_exponential(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        exponential(scale, num).to_pyarray(py)
    }

    #[pyfn(m, "laplace")]
    fn py_laplace(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the laplace function. Converts
        /// the rust vector into a numpy array
        laplace(scale, num).to_pyarray(py)
    }

    #[pyfn(m, "geometric")]
    fn py_geometric(py: Python, param: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the geometric function. Converts
        /// the rust vector into a numpy array
        geometric(param, num).to_pyarray(py)
    }

    #[pyfn(m, "two_sided_geometric")]
    fn py_two_sided_geometric(py: Python, param: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the two sided geometric function. Converts
        /// the rust vector into a numpy array
        two_sided_geometric(param, num).to_pyarray(py)
    }

    Ok(())
}