//! This module contains the `Tensor` struct, which is a multi-dimensional array that generalizes
//! vectors and matrices to potentially higher dimensions.

use num_traits::{One, Zero};

pub use shape::Shape;

mod cast;
mod ops;
#[cfg(feature = "random")]
mod random;
mod reshape;
mod shape;

/// A multi-dimensional array that generalizes vectors and matrices to potentially higher
/// dimensions.
#[derive(Debug, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

// Implementing basic methods for Tensor<T>.
impl<T> Tensor<T> {
    /// Create a new tensor from the given data and shape.
    pub fn new<U, S>(data: U, shape: S) -> Self
    where
        U: Into<Vec<T>>,
        S: Into<Shape>,
    {
        let data = data.into();
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data does not match shape size.");
        Self { data, shape }
    }

    /// Return a reference to the raw underlying data of the tensor.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Return the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }
}

// Implementing constructors for Tensor<T>.
impl<T: Clone + Zero + One> Tensor<T> {
    /// Create a new tensor of zeros with the given shape.
    pub fn zeros<S>(shape: S) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let numel = shape.numel();
        let data = vec![T::zero(); numel];
        Self { data, shape }
    }

    /// Create a new tensor of ones with the given shape.
    pub fn ones<S>(shape: S) -> Self
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let size = shape.numel();
        let data = vec![T::one(); size];
        Self { data, shape }
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
        assert_eq!(tensor.data(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_zeros() {
        let shape = [2, 2];
        let tensor: Tensor<u32> = Tensor::zeros(shape);
        assert_eq!(tensor.data(), vec![0, 0, 0, 0]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_ones() {
        let shape = vec![2, 2];
        let tensor: Tensor<u32> = Tensor::ones(shape);
        assert_eq!(tensor.data(), vec![1, 1, 1, 1]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_eq() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        assert_eq!(tensor1, tensor2);
    }

    #[test]
    fn test_ne() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        assert_ne!(tensor1, tensor2);
    }
}
