use std::borrow::Borrow;

use crate::{Entry, Index};

pub struct View<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
}

impl<'a, T> View<'a, T> {
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Self {
        Self { data, shape }
    }
}

impl<'a, T: Clone> Index for View<'a, T> {
    type Output<'b> = Entry<T, View<'a, T>> where Self: 'b;
    fn index(&self, index: usize) -> Self::Output<'_> {
        if self.shape.len() == 1 {
            Entry::Scalar(self.data[index].clone())
        } else {
            let stride = self.shape[1..].iter().product::<usize>();
            let start = index * stride;
            let end = start + stride;
            Entry::Slice(View {
                data: &self.data[start..end],
                shape: &self.shape[1..],
            })
        }
    }
}

impl<'a, T> Borrow<[T]> for View<'a, T> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use crate::Convert;

    use super::*;

    #[test]
    fn test_index() {
        let tensor = [1, 2, 3, 4].into_vector();
        let view = tensor.view();
        assert_eq!(view.index(0).scalar(), 1);
        assert_eq!(view.index(1).scalar(), 2);
        assert_eq!(view.index(2).scalar(), 3);
        assert_eq!(view.index(3).scalar(), 4);
    }

    #[test]
    fn test_index_2d() {
        let tensor = [1, 2, 3, 4].into_tensor([2, 2]);
        let view = tensor.view();
        assert_eq!(view.index(0).slice(), &[1, 2]);
        assert_eq!(view.index(1).slice(), &[3, 4]);
    }
}
