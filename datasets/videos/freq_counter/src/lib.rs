use pyo3::prelude::*;
use std::collections::HashMap;
use std::cmp::Reverse;

#[pyclass]
pub struct FreqCounter {
    term_freqs: HashMap<String, i64>,
    doc_freqs: HashMap<String, i64>,
}

#[pymethods]
impl FreqCounter {
    #[new]
    fn new() -> Self {
        FreqCounter {
            term_freqs: HashMap::new(),
            doc_freqs: HashMap::new(),
        }
    }

    fn count_tokens_freq(&mut self, tokens: Vec<String>) {
        for token in tokens {
            *self.term_freqs.entry(token.clone()).or_insert(0) += 1;
            self.doc_freqs.insert(token, 1);
        }
    }

    fn sort_freqs(&mut self) {
        let mut term_vec: Vec<_> = self.term_freqs.clone().into_iter().collect();
        let mut doc_vec: Vec<_> = self.doc_freqs.clone().into_iter().collect();

        term_vec.sort_by_key(|x| Reverse(x.1));
        doc_vec.sort_by_key(|x| Reverse(x.1));

        self.term_freqs = term_vec.into_iter().collect();
        self.doc_freqs = doc_vec.into_iter().collect();
    }

    #[getter]
    fn get_term_freqs(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.term_freqs {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    #[getter]
    fn get_doc_freqs(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.doc_freqs {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }
}

#[pymodule]
fn freq_counter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FreqCounter>()?;
    Ok(())
}