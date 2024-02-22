//! This module contains the `Tensor` struct, which is a multi-dimensional array that generalizes
//! vectors and matrices to potentially higher dimensions.

use num_traits::{One, Zero};

/// A multi-dimensional array that generalizes vectors and matrices to potentially higher
/// dimensions.
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone + Zero + One> Tensor<T> {
    /// Create a new tensor from the given data and shape.
    pub fn new<U, S>(data: U, shape: S) -> Self
        where
            U: Into<Vec<T>>,
            S: Into<Vec<usize>>,
    {
        let data = data.into();
        let shape = shape.into();
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Data does not match shape size."
        );
        Self { data, shape }
    }

    /// Create a new tensor of zeros with the given shape.
    pub fn zeros<S>(shape: S) -> Self
        where
            S: Into<Vec<usize>>,
    {
        let shape = shape.into();
        let size = shape.iter().product();
        let data = vec![T::zero(); size];
        Self { data, shape }
    }

    /// Create a new tensor of ones with the given shape.
    pub fn ones<S>(shape: S) -> Self
        where
            S: Into<Vec<usize>>,
    {
        let shape = shape.into();
        let size = shape.iter().product();
        let data = vec![T::one(); size];
        Self { data, shape }
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = vec![1, 2, 3, 4];
        let shape = [2, 2];
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.data, vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_zeros() {
        let shape = [2, 2];
        let tensor: Tensor<u32> = Tensor::zeros(shape);
        assert_eq!(tensor.data, vec![0, 0, 0, 0]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_ones() {
        let shape = vec![2, 2];
        let tensor: Tensor<u32> = Tensor::ones(shape);
        assert_eq!(tensor.data, vec![1, 1, 1, 1]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }
}
