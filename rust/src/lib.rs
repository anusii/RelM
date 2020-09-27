use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::prelude::*;
use rayon::prelude::*;


#[pyfunction]
fn py_laplace(scale: f64, num: usize) -> PyResult<Vec<f64>> {
    Ok(laplace(scale, num))
}


fn laplace(scale: f64, num: usize) -> Vec<f64> {
    let mut samples: Vec<f64> = vec![scale; num];
    samples.par_iter_mut().for_each(|p| *p = scalar_laplace(*p));
    samples
}


fn scalar_laplace(scale: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let y = rng.gen::<f64>() - 0.5;
    let sgn = y.signum();
    sgn * (2.0 * sgn * y).ln() * scale
}


/// A Python module implemented in Rust.
#[pymodule]
fn primitives(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_laplace, m)?)?;
    Ok(())
}