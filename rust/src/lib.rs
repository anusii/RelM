use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rand::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray, PyArray1, ToPyArray};


fn scalar_laplace(scale: f64) -> f64 {
    /// Returns one sample from the Laplace distribution
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter of the Laplace distribution
    let mut rng = rand::thread_rng();
    let y = rng.gen::<f64>() - 0.5;
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


///// A Python module implemented in Rust.
//// Exports the rust functions to python.
#[pymodule]
fn primitives(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "laplace")]
    fn py_laplace(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the laplace function. Converts
        /// the rust vector into a numpy array
        laplace(scale, num).to_pyarray(py)
    }

    Ok(())
}