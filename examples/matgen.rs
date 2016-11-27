//! # `matgen` example

extern crate linxal;
extern crate ndarray;
extern crate rand;

use rand::thread_rng;
use linxal::generate::{RandomGeneral, RandomSemiPositive};

fn generate_general() {
    let mut rng = thread_rng();

    let mut generator: RandomGeneral<f32> =  RandomGeneral::new(3, 3, &mut rng);

    generator.rank(2).bands(1, 2);
    println!("Rank 2, (1, 2) - banded matrix");
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());
    println!("---");
    generator.full_rank().full_bands();
    println!("Full rank, full-banded matrix");
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());
    println!("---");
    generator.bands(0, 0).sv_random_uniform(1.0, 4.0);
    println!("Diagnoal matrix with random singular values");
    println!("{:?}", generator.generate().unwrap());
    let (g, v) = generator.generate_with_sv().unwrap();
    println!("{:?}", g);
    println!("{:?}", v);
}


fn generate_positive() {
    let mut rng = thread_rng();

    let mut generator: RandomSemiPositive<f32> =  RandomSemiPositive::new(4, &mut rng);

    generator.bands(2);
    println!("Positive, (2, 2)-banded matrix");
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());

    generator.full_bands();
    println!("fully-banded matix");
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());


    generator.rank(2);
    println!("Rank 2 matrix");
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());
}

fn main() {
    generate_general();
    generate_positive();
}
