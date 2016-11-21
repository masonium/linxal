//! # `matgen` example

extern crate linxal;
extern crate ndarray;
extern crate rand;

use rand::thread_rng;
use linxal::generate::{Packing, RandomGeneral};

fn main() {
    let mut rng = thread_rng();

    let mut generator: RandomGeneral<f32> =  RandomGeneral::new(3, 3, &mut rng);

    generator.rank(2).bands(1, 2);

    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());

    generator.full_rank().full_bands();
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());

    generator.bands(0, 0).sv_random_uniform(1.0, 4.0);
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());

    generator.rank(2);
    println!("{:?}", generator.generate().unwrap());
    println!("{:?}", generator.generate().unwrap());
}
