//! This module contains the implementation of the `cast` method for the `Tensor` struct.

use super::Tensor;

impl<T> Tensor<T> {
    /// Cast the raw to a new type.
    pub fn cast<U: From<T>>(self) -> Tensor<U> {
        Tensor::new(
            self.data
                .into_iter()
                .map(|x| U::from(x))
                .collect::<Vec<_>>(),
            self.shape,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor = tensor.cast::<f64>();
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
