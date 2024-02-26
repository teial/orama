use std::borrow::Borrow;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Entry<E, S> {
    Scalar(E),
    Slice(S),
}

impl<E, S> Entry<E, S>
where
    E: Clone,
    S: Borrow<[E]>,
{
    pub fn scalar(&self) -> E {
        match self {
            Entry::Scalar(e) => e.clone(),
            Entry::Slice(_) => panic!("Cannot get element from slice."),
        }
    }

    pub fn slice(&self) -> &[E] {
        match self {
            Entry::Scalar(_) => panic!("Cannot get slice from element."),
            Entry::Slice(s) => s.borrow(),
        }
    }
}

pub trait Index {
    type Output<'a>
    where
        Self: 'a;
    fn index(&self, index: usize) -> Self::Output<'_>;
}
