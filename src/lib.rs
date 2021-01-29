#![allow(unused_doc_comments)]

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::{PyArray1, ToPyArray};


mod utils;
mod samplers;
mod mechanisms;


///// A Python module implemented in Rust.
///// Exports the rust functions to python.
#[pymodule]
fn backend(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "all_above_threshold")]
    fn py_all_above_threshold<'a>(
        py: Python<'a>,
        data: &'a PyArray1<f64>,
        epsilon: f64,
        threshold: f64,
        precision: i32,
    ) -> &'a PyArray1<usize> {
        /// Simple python wrapper of the exponential function. Converts
        /// the rust vector into a numpy array
        let data = data.to_vec().unwrap();
        mechanisms::all_above_threshold(data, epsilon, threshold, precision).to_pyarray(py)
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
        epsilon: f64,
        precision: i32,
    ) -> &'a PyArray1<f64> {
        let data = data.to_vec().unwrap();
        mechanisms::laplace_mechanism(data, epsilon, precision).to_pyarray(py)
    }

    #[pyfn(m, "geometric_mechanism")]
    fn py_geometric_mechanism<'a>(
        py: Python<'a>,
        data: &'a PyArray1<i64>,
        epsilon: f64,
    ) -> &'a PyArray1<i64> {
        let data = data.to_vec().unwrap();
        mechanisms::geometric_mechanism(data, epsilon).to_pyarray(py)
    }


    #[pyfn(m, "exponential_mechanism_weighted_index")]
    fn py_exponential_mechanism_weighted_index<'a>(
        py: Python<'a>,
        utilities: &'a PyArray1<f64>,
        epsilon: f64,
    ) -> PyResult<u64> {
        let utilities = utilities.to_vec().unwrap();
        let index: u64 = mechanisms::exponential_mechanism_weighted_index(
            utilities,
            epsilon,
        );
        Ok(index)
    }


    #[pyfn(m, "exponential_mechanism_gumbel_trick")]
    fn py_exponential_mechanism_gumbel_trick<'a>(
        py: Python<'a>,
        utilities: &'a PyArray1<f64>,
        epsilon: f64,
    ) -> PyResult<u64> {
        let utilities = utilities.to_vec().unwrap();
        let index: u64 = mechanisms::exponential_mechanism_gumbel_trick(
            utilities,
            epsilon,
        );
        Ok(index)
    }


    #[pyfn(m, "exponential_mechanism_sample_and_flip")]
    fn py_exponential_mechanism_sample_and_flip<'a>(
        py: Python<'a>,
        utilities: &'a PyArray1<f64>,
        epsilon: f64,
    ) -> PyResult<u64> {
        let utilities = utilities.to_vec().unwrap();
        let index: u64 = mechanisms::exponential_mechanism_sample_and_flip(
            utilities,
            epsilon,
        );
        Ok(index)
    }


    #[pyfn(m, "permute_and_flip_mechanism")]
    fn py_permute_and_flip_mechanism<'a>(
        py: Python<'a>,
        utilities: &'a PyArray1<f64>,
        epsilon: f64,
    ) -> PyResult<u64> {
        let utilities = utilities.to_vec().unwrap();
        let index: u64 = mechanisms::permute_and_flip_mechanism(
            utilities,
            epsilon,
        );
        Ok(index)
    }

    #[pyfn(m, "small_db")]
    fn py_small_db<'a>(
        py: Python<'a>,
        epsilon: f64,
        l1_norm: usize,
        size: u64,
        db_l1_norm: u64,
        queries: &'a PyArray1<u64>,
        answers: &'a PyArray1<f64>,
        breaks: &'a PyArray1<u64>
) -> &'a PyArray1<u64> {
        let queries = queries.to_vec().unwrap();
        let answers = answers.to_vec().unwrap();
        let breaks = breaks.to_vec().unwrap();
        let breaks = breaks.iter().map(|&x| x as usize).collect();

        mechanisms::small_db(epsilon, l1_norm, size, db_l1_norm, queries, answers, breaks).to_pyarray(py)
    }

    Ok(())
}
