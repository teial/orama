//! This module contains the implementation of the arithmetic operations for the Tensor struct.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Tensor;

// Implementing AddAssign trait for Tensor<T>
#[auto_impl_ops::auto_ops]
impl<T: AddAssign + Copy> AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape to be added"
        );
        self.data
            .iter_mut()
            .zip(other.data().iter())
            .for_each(|(a, b)| *a += *b);
    }
}

// Implementing SubAssign trait for Tensor<T>
#[auto_impl_ops::auto_ops]
impl<T: SubAssign + Copy> SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape to be subtracted"
        );
        self.data
            .iter_mut()
            .zip(other.data().iter())
            .for_each(|(a, b)| *a -= *b);
    }
}

// Implementing MulAssign trait for Tensor<T>
#[auto_impl_ops::auto_ops]
impl<T: MulAssign + Copy> MulAssign<&Tensor<T>> for Tensor<T> {
    fn mul_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape to be multiplied"
        );
        self.data
            .iter_mut()
            .zip(other.data().iter())
            .for_each(|(a, b)| *a *= *b);
    }
}

// Implementing DivAssign trait for Tensor<T>
#[auto_impl_ops::auto_ops]
impl<T: DivAssign + Copy> DivAssign<&Tensor<T>> for Tensor<T> {
    fn div_assign(&mut self, other: &Self) {
        assert_eq!(
            self.shape, other.shape,
            "Tensors must have the same shape to be divided"
        );
        self.data
            .iter_mut()
            .zip(other.data().iter())
            .for_each(|(a, b)| *a /= *b);
    }
}

// Implementing AddAssign trait for Tensor<T> with scalar T
#[auto_impl_ops::auto_ops]
impl<T: AddAssign + Copy> AddAssign<T> for Tensor<T> {
    fn add_assign(&mut self, other: T) {
        self.data.iter_mut().for_each(|a| *a += other);
    }
}

// Implementing SubAssign trait for Tensor<T> with scalar T
#[auto_impl_ops::auto_ops]
impl<T: SubAssign + Copy> SubAssign<T> for Tensor<T> {
    fn sub_assign(&mut self, other: T) {
        self.data.iter_mut().for_each(|a| *a -= other);
    }
}

// Implementing MulAssign trait for Tensor<T> with scalar T
#[auto_impl_ops::auto_ops]
impl<T: MulAssign + Copy> MulAssign<T> for Tensor<T> {
    fn mul_assign(&mut self, other: T) {
        self.data.iter_mut().for_each(|a| *a *= other);
    }
}

// Implementing DivAssign trait for Tensor<T> with scalar T
#[auto_impl_ops::auto_ops]
impl<T: DivAssign + Copy> DivAssign<T> for Tensor<T> {
    fn div_assign(&mut self, other: T) {
        self.data.iter_mut().for_each(|a| *a /= other);
    }
}

// Implementing Neg trait for Tensor<T>
impl<T: Neg<Output = T> + Copy> Neg for Tensor<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.data.iter_mut().for_each(|a| *a = -*a);
        self
    }
}

#[cfg(test)]
mod tests {
    use approx::*;

    use super::*;

    #[test]
    fn test_add() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let result = tensor1 + &tensor2;
        assert_eq!(result, Tensor::new(vec![6, 8, 10, 12], vec![2, 2]));
    }

    #[test]
    fn test_sub() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let result = tensor1 - &tensor2;
        assert_eq!(result, Tensor::new(vec![-4, -4, -4, -4], vec![2, 2]));
    }

    #[test]
    fn test_mul() {
        let tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let result = tensor1 * &tensor2;
        assert_eq!(result, Tensor::new(vec![5, 12, 21, 32], vec![2, 2]));
    }

    #[test]
    fn test_div() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = tensor1 / &tensor2;
        assert_ulps_eq!(
            result.data(),
            Tensor::new(vec![0.2, 0.33333, 0.42857, 0.5], vec![2, 2]).data(),
            epsilon = 1e-5,
        );
    }

    #[test]
    fn test_add_assign() {
        let mut tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        tensor1 += &tensor2;
        assert_eq!(tensor1, Tensor::new(vec![6, 8, 10, 12], vec![2, 2]));
    }

    #[test]
    fn test_sub_assign() {
        let mut tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        tensor1 -= &tensor2;
        assert_eq!(tensor1, Tensor::new(vec![-4, -4, -4, -4], vec![2, 2]));
    }

    #[test]
    fn test_mul_assign() {
        let mut tensor1 = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        tensor1 *= &tensor2;
        assert_eq!(tensor1, Tensor::new(vec![5, 12, 21, 32], vec![2, 2]));
    }

    #[test]
    fn test_div_assign() {
        let mut tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        tensor1 /= &tensor2;
        assert_ulps_eq!(
            tensor1.data(),
            Tensor::new(vec![0.2, 0.33333, 0.42857, 0.5], vec![2, 2]).data(),
            epsilon = 1e-5,
        );
    }

    #[test]
    fn test_add_scalar() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let result = tensor + 1;
        assert_eq!(result, Tensor::new(vec![2, 3, 4, 5], vec![2, 2]));
    }

    #[test]
    fn test_sub_scalar() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let result = tensor - 1;
        assert_eq!(result, Tensor::new(vec![0, 1, 2, 3], vec![2, 2]));
    }

    #[test]
    fn test_mul_scalar() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let result = tensor * 2;
        assert_eq!(result, Tensor::new(vec![2, 4, 6, 8], vec![2, 2]));
    }

    #[test]
    fn test_div_scalar() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor / 2.0;
        assert_ulps_eq!(
            result.data(),
            Tensor::new(vec![0.5, 1.0, 1.5, 2.0], vec![2, 2]).data()
        );
    }

    #[test]
    fn test_add_assign_scalar() {
        let mut tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        tensor += 1;
        assert_eq!(tensor, Tensor::new(vec![2, 3, 4, 5], vec![2, 2]));
    }

    #[test]
    fn test_sub_assign_scalar() {
        let mut tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        tensor -= 1;
        assert_eq!(tensor, Tensor::new(vec![0, 1, 2, 3], vec![2, 2]));
    }

    #[test]
    fn test_mul_assign_scalar() {
        let mut tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        tensor *= 2;
        assert_eq!(tensor, Tensor::new(vec![2, 4, 6, 8], vec![2, 2]));
    }

    #[test]
    fn test_div_assign_scalar() {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tensor /= 2.0;
        assert_ulps_eq!(
            tensor.data(),
            Tensor::new(vec![0.5, 1.0, 1.5, 2.0], vec![2, 2]).data()
        );
    }

    #[test]
    fn test_neg() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let result = -tensor;
        assert_eq!(result, Tensor::new(vec![-1, -2, -3, -4], vec![2, 2]));
    }
}
