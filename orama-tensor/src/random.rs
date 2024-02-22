//! Random number generation and distribution methods for tensors.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use rand_distr::{Normal, Uniform};

use crate::Tensor;

impl<T> Tensor<T> {
    /// Create a new tensor from the given shape and distribution.
    pub fn from_distribution<D>(shape: Vec<usize>, distribution: D, rng: &mut impl Rng) -> Self
        where
            D: Distribution<T>,
    {
        let size = shape.iter().product();
        let data: Vec<_> = distribution.sample_iter(rng).take(size).collect();
        Self::new(data, shape)
    }

    /// Convenience method for creating a tensor from a uniform distribution.
    /// The low and high parameters are inclusive. If low > high, or if the parameters are
    /// not finite, the method panics.
    pub fn from_uniform(shape: Vec<usize>, low: T, high: T, rng: &mut impl Rng) -> Self
        where
            T: rand::distributions::uniform::SampleUniform,
    {
        let distribution =
            Uniform::new_inclusive(low, high).expect("Invalid parameters for uniform distribution");
        Self::from_distribution(shape, distribution, rng)
    }
}

impl<T> Tensor<T>
    where
        StandardNormal: Distribution<T>,
{
    /// Convenience method for creating a tensor from a normal distribution.
    /// The mean and standard deviation parameters are used to create the distribution.
    /// If the standard deviation is not finite, the method panics.
    pub fn from_normal(shape: Vec<usize>, mean: T, std_dev: T, rng: &mut impl Rng) -> Self
        where
            T: num_traits::Float,
    {
        let distribution =
            Normal::new(mean, std_dev).expect("Invalid parameters for normal distribution");
        Self::from_distribution(shape, distribution, rng)
    }

    /// Convenience method for creating a tensor from a standard normal distribution.
    pub fn from_standard_normal(shape: Vec<usize>, rng: &mut impl Rng) -> Self {
        let distribution = StandardNormal;
        Self::from_distribution(shape, distribution, rng)
    }
}

#[cfg(test)]
mod tests {
    use approx::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::Bernoulli;

    use super::*;

    #[test]
    fn test_from_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tensor = Tensor::from_uniform(vec![2, 3], 0.0, 1.0, &mut rng);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_ulps_eq!(
            tensor.data(),
            [0.68189, 0.95027, 0.42751, 0.62736, 0.28859, 0.14995].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    #[should_panic]
    fn test_from_uniform_invalid_inverse() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        Tensor::from_uniform(vec![2, 3], 1.0, 0.0, &mut rng);
    }

    #[test]
    #[should_panic]
    fn test_from_uniform_invalid_nan() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        Tensor::from_uniform(vec![2, 3], 0.0, f64::INFINITY, &mut rng);
    }

    #[test]
    fn test_from_normal() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tensor = Tensor::from_normal(vec![2, 3], 0.0, 1.0, &mut rng);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_ulps_eq!(
            tensor.data(),
            [0.47798, 1.33407, -0.21086, 0.47634, -0.51209, -0.93397].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    #[should_panic]
    fn test_from_normal_invalid() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        Tensor::from_normal(vec![2, 3], 0.0, f64::INFINITY, &mut rng);
    }

    #[test]
    fn test_from_standard_normal() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tensor = Tensor::from_standard_normal(vec![2, 3], &mut rng);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_ulps_eq!(
            tensor.data(),
            [0.47798, 1.33407, -0.21086, 0.47634, -0.51209, -0.93397].as_slice(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_from_distribution() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let distribution = Bernoulli::new(0.5).unwrap();
        let tensor = Tensor::from_distribution(vec![2, 3], distribution, &mut rng);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(tensor.data(), [false, false, true, false, true, true]);
    }
}
