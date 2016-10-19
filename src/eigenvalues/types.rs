pub enum EigenError {
    Success,
    BadInput,
    BadParameter(i32),
    Failed
}

pub enum ComputeVectors {
    Left,
    Right,
    Both
}
