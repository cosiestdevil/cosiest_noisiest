use cosiest_noisiest::NoiseGenerator;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    // Run registered benchmarks.
    divan::main();
}
const SIZE: usize = 256;
#[divan::bench(args = [true, false],sample_count=1000,sample_size=100,threads=8)]
fn simple(fill: bool) -> [f64;SIZE] {
    let mut noise_generator =
        NoiseGenerator::from_rng(ChaCha20Rng::seed_from_u64(2), 1. / 32., 128., 3);
    
    let mut result = [0.0; SIZE];
    if fill {
        noise_generator.fill(0, &mut result);
    } else {
        for (x, y) in result.iter_mut().enumerate().take(SIZE) {
            *y = noise_generator.sample(x);
        }
    }
    result
}
