use cosiest_noisiest::*;

#[doc(hidden)]
#[allow(clippy::needless_range_loop)]
fn main() {
    let mut noise_generator =
        NoiseGenerator::from_rng(ChaCha20Rng::seed_from_u64(2), Frequency::from_wave_length(32), 128., 9);
    const SIZE: usize = 5;
    let mut temp = [0.0; SIZE*4];
    noise_generator.fill(0, &mut temp);
    for x in 0..SIZE*2 {
        let y = noise_generator.sample(x);
        println!("{}: {} == {} = {}", x, temp[x], y, temp[x] == y);
    }
    let mut temp = [0.0; SIZE];
    noise_generator.fill(7, &mut temp);
    for x in 0..SIZE {
        let y = noise_generator.sample(x+7);
        println!("{}: {} == {} = {}", x+7, temp[x], y, temp[x] == y);
    }
    
}
