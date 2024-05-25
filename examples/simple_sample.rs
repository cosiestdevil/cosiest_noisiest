use cosiest_noisiest::*;

#[doc(hidden)]
#[allow(clippy::needless_range_loop)]
fn main() {
    let mut noise_generator =
        NoiseGenerator::from_rng(ChaCha20Rng::seed_from_u64(2), Frequency::from_wave_length(32), 128., 3);
    let mut result = [0.; 512];

    for x in 0..256 {
        let y = noise_generator.sample(x);
        result[x] = y
    }
    for x in (256..512).rev() {
        let y = noise_generator.sample(x);
        result[x] = y
    }
    for (x,y) in result.into_iter().enumerate() {
        println!("{}: {}",x, y);
    }
}
