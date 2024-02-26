use crate::{Shape, Tensor};

pub trait Convert<T>
    where
        Self: Sized,
{
    fn length(&self) -> usize;

    fn into_tensor(self, shape: impl Into<Shape>) -> Tensor<T>;

    fn into_vector(self) -> Tensor<T> {
        let length = self.length();
        self.into_tensor([length])
    }

    fn into_matrix(self, rows: usize, cols: usize) -> Tensor<T> {
        self.into_tensor([rows, cols])
    }
}

impl<T> Convert<T> for Vec<T>
    where
        T: Clone,
{
    fn length(&self) -> usize {
        self.len()
    }

    fn into_tensor(self, shape: impl Into<Shape>) -> Tensor<T> {
        Tensor::new(self, shape)
    }
}

impl<T> Convert<T> for Tensor<T>
    where
        T: Clone,
{
    fn length(&self) -> usize {
        self.numel()
    }

    fn into_tensor(self, shape: impl Into<Shape>) -> Tensor<T> {
        let shape = shape.into();
        assert_eq!(self.numel(), shape.numel());
        self
    }
}

impl<T: Clone> Convert<T> for &[T] {
    fn length(&self) -> usize {
        self.len()
    }

    fn into_tensor(self, shape: impl Into<Shape>) -> Tensor<T> {
        let shape = shape.into();
        assert_eq!(self.len(), shape.numel());
        Tensor::new(self, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_tensor() {
        let tensor: Tensor<u32> = vec![1, 2, 3, 4].into_tensor([2, 2]);
        assert_eq!(tensor.data(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_into_vector() {
        let tensor: Tensor<u32> = vec![1, 2, 3, 4].into_vector();
        assert_eq!(tensor.data(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![4]);
    }

    #[test]
    fn test_into_matrix() {
        let tensor: Tensor<u32> = vec![1, 2, 3, 4].into_matrix(2, 2);
        assert_eq!(tensor.data(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_into_tensor_ref() {
        let data = vec![1, 2, 3, 4];
        let tensor: Tensor<u32> = data.as_slice().into_tensor([2, 2]);
        assert_eq!(tensor.data(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.shape(), vec![2, 2]);
    }
}
