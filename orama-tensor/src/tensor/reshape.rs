use super::{Shape, Tensor};

struct Spec(Vec<usize>);

impl<T: Into<Vec<usize>>> From<T> for Spec {
    fn from(shape: T) -> Self {
        let shape = shape.into();
        assert!(
            shape.iter().filter(|&&dim| dim == 0).count() <= 1,
            "Shape specification should have at most one zero dimension."
        );
        Self(shape)
    }
}

impl Spec {
    fn product(&self) -> usize {
        self.0.iter().filter(|&&dim| dim != 0).product()
    }

    fn into_shape(mut self, numel: usize) -> Result<Shape, String> {
        if let Some(zero_index) = self.0.iter().position(|&dim| dim == 0) {
            let new_dim = numel / self.product();
            self.0[zero_index] = new_dim;
        }
        Ok(self.0.into())
    }
}

// Implementing reshaping methods for Tensor<T>.
impl<T> Tensor<T> {
    pub fn reshape(mut self, new_shape: impl Into<Vec<usize>>) -> Result<Self, String> {
        let new_shape = Spec::from(new_shape).into_shape(self.data.len())?;
        self.shape = new_shape;
        Ok(self)
    }
}
