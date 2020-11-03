use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::{PyArray, PyArray1, ToPyArray};
use rayon::prelude::*;
use rug::Float;


mod utils;
mod samplers;
mod mechanisms;

///// A Python module implemented in Rust.
///// Exports the rust functions to python.
#[pymodule]
fn backend(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "uniform")]
    fn py_uniform(py: Python, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array

        utils::vectorize(1.0, num, samplers::uniform).to_pyarray(py)
    }

    #[pyfn(m, "exponential")]
    fn py_exponential(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array

        utils::vectorize(scale, num, samplers::exponential).to_pyarray(py)
    }

    #[pyfn(m, "laplace")]
    fn py_laplace(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the laplace function. Converts
        /// the rust vector into a numpy array

        utils::vectorize(scale, num, samplers::laplace).to_pyarray(py)
    }

    #[pyfn(m, "geometric")]
    fn py_geometric(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the geometric function. Converts
        /// the rust vector into a numpy array

        utils::vectorize(scale, num, samplers::geometric).to_pyarray(py)
    }

    #[pyfn(m, "two_sided_geometric")]
    fn py_two_sided_geometric(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the two sided geometric function. Converts
        /// the rust vector into a numpy array

        utils::vectorize(scale, num, samplers::two_sided_geometric).to_pyarray(py)
    }

    #[pyfn(m, "double_uniform")]
    fn py_double_uniform(py: Python, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        utils::vectorize(1.0, num, samplers::double_uniform).to_pyarray(py)
    }

    #[pyfn(m, "all_above_threshold")]
    fn py_all_above_threshold<'a>(
        py: Python<'a>, data: &'a PyArray1<f64>,
        scale: f64, threshold: f64
    ) -> &'a PyArray1<usize> {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let data = data.to_vec().unwrap();
        mechanisms::all_above_threshold(data, scale, threshold).to_pyarray(py)
    }

    #[pyfn(m, "snapping")]
    fn py_snapping<'a>(
        py: Python<'a>, data: &'a PyArray1<f64>,
        bound: f64, lambda: f64, quanta: f64
    ) -> &'a PyArray1<f64> {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let data = data.to_vec().unwrap();
        mechanisms::snapping(data, bound, lambda, quanta).to_pyarray(py)
    }

    #[pyfn(m, "ln_rn")]
    fn py_ln_rn(x: f64) -> f64 {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        utils::ln_rn(x)
    }


    #[pyfn(m, "fixed_point_laplace")]
    fn py_fixed_point_laplace(py: Python, scale: f64, num: usize) -> &PyArray1<f64>{
        /// Simple python wrapper of the laplace function. Converts
        /// the rust vector into a numpy array
        let num_bits = 58;
        let mut biases: Vec<u64> = vec![0; 64];
        for idx in 0..64 {
            let d = Float::with_val(num_bits, 2.0f64.powi(32 - idx)) / Float::with_val(num_bits, scale);
            let bias = Float::with_val(num_bits, 1.0) / (Float::with_val(num_bits, 1.0) + d.exp());
            let bias = (bias * Float::with_val(num_bits, 2.0f64.powi(64))).to_f64() as u64;
            biases[idx as usize] = bias;
        }

        let mut samples: Vec<f64> = vec![0.0; num];
        samples.par_iter_mut().for_each(|p| *p = samplers::fixed_point_laplace(&biases));
        samples.to_pyarray(py)
    }

    Ok(())

}