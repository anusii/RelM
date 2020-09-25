use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::prelude::*;
use rayon::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn laplace(scale: f64, num: usize) -> PyResult<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut samples: Vec<f64> = vec![0.0; num];
    for i in 0..num {
        let y = rng.gen::<f64>() - 0.5;
        let sgn = y.signum();
        samples[i] = sgn * (2.0 * sgn * y).ln() * scale;
    }

    Ok(samples)
}



/// A Python module implemented in Rust.
#[pymodule]
fn primitives(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(laplace, m)?)?;
    Ok(())
}