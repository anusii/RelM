#![allow(unused_doc_comments)]

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::{PyArray1, ToPyArray};


mod utils;
mod samplers;
mod mechanisms;


///// A Python module implemented in Rust.
///// Exports the rust functions to python.
#[pymodule]
fn backend(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "all_above_threshold")]
    fn py_all_above_threshold<'a>(
        py: Python<'a>,
        data: &'a PyArray1<f64>,
        scale: f64,
        threshold: f64,
        precision: i32,
    ) -> &'a PyArray1<usize> {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let data = data.to_vec().unwrap();
        mechanisms::all_above_threshold(data, scale, threshold, precision).to_pyarray(py)
    }

    #[pyfn(m, "snapping")]
    fn py_snapping<'a>(
        py: Python<'a>,
        data: &'a PyArray1<f64>,
        bound: f64,
        lambda: f64,
        quanta: f64,
    ) -> &'a PyArray1<f64> {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let data = data.to_vec().unwrap();
        mechanisms::snapping(data, bound, lambda, quanta).to_pyarray(py)
    }

    #[pyfn(m, "laplace_mechanism")]
    fn py_laplace_mechanism<'a>(
        py: Python<'a>,
        data: &'a PyArray1<f64>,
        sensitivity: f64,
        epsilon: f64,
        precision: i32,
    ) -> &'a PyArray1<f64> {
        let data = data.to_vec().unwrap();
        mechanisms::laplace_mechanism(data, sensitivity, epsilon, precision).to_pyarray(py)
    }

    Ok(())
}
