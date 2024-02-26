#[derive(Debug, PartialEq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn size(&self) -> usize {
        self.0.len()
    }
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl<T: Into<Vec<usize>>> From<T> for Shape {
    fn from(shape: T) -> Self {
        let shape = shape.into();
        assert!(
            shape.iter().any(|&dim| dim != 0),
            "Shape dimensions cannot be zero."
        );
        Self(shape)
    }
}
