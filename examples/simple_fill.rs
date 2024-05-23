use cosiest_noisiest::NoiseGenerator;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[doc(hidden)]
#[allow(clippy::needless_range_loop)]
fn main() {
    let mut noise_generator =
        NoiseGenerator::from_rng(ChaCha20Rng::seed_from_u64(2), 1. / 32., 128., 3);
    const SIZE: usize = 256;
    let mut temp = [0.0; SIZE];
    noise_generator.fill(0, &mut temp);
    for x in 0..SIZE {
        let y = noise_generator.sample(x);
        println!("{}: {} == {} = {}", x, temp[x], y, temp[x] == y);
    }
}
