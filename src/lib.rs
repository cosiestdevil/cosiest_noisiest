#![deny(missing_docs)]
//! # Cosiest Noisiest
//!
//! `cosiest_noisiest` allows generating 1d noise at arbitrary points
//! # Examples
//! ```
//! use cosiest_noisiest::NoiseGenerator;
//!
//! let mut noise_generator = NoiseGenerator::from_u64_seed(2, 1. / 32., 128., 3);
//! let noise:f64 = noise_generator.sample(123456789);
//! ```
//!
//! # Crate Features
//! **f32** - Enables using f32 when interpolating between noise values instead of the default f64.
//!

use cfg_if::cfg_if;
use num_integer::gcd;
use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use splines::{Interpolate, Interpolation, Key, Spline};
use std::mem::size_of;
use std::ops::{AddAssign, Mul};

cfg_if! {
    if #[cfg(all(feature="f32"))]{
        type Interpolator = f32;
    }else{
        type Interpolator = f64;
    }
}

/// A 1D Noise Generator that allows sampling at arbitrary points.
///
/// # Examples
/// ```
/// use cosiest_noisiest::NoiseGenerator;
///
/// let mut noise_generator = NoiseGenerator::from_u64_seed(2, 1. / 32., 128., 3);
/// let noise:f64 = noise_generator.sample(123456789);
/// ```
///
/// Only supports ChaCha20 as the RNG source. This is in part because ChaCha based RNG allows setting the word position of the stream, allowing arbitrary sampling.
#[derive(Clone, Debug)]
pub struct NoiseGenerator<
    T: WordAligned
        + Default
        + Interpolate<Interpolator>
        + AddAssign
        + Mul<f64, Output = T>
        + std::fmt::Debug,
> where
    Standard: Distribution<T>,
{
    rng: ChaCha20Rng,
    /// The frequency of the noise used to calculate the wavelength of noise to generate.
    pub frequency: f64,
    /// The amount of words to offset the RNG source, determined from the size of the generation result
    offset_size: usize,
    /// The amplitude of the noise
    pub amplitude: f64,
    /// How many octaves of noise to generate
    ///
    /// If this is coprime with the resulting wavelength (based on the frequency) then an extra octave is generated and each octave will be half as large as the previous.
    ///
    /// If this is not coprime with the wavelength then each octave will be `1/gcd(wavelength,octaves)`` as large as the previous.
    pub octaves: usize,
    current_spline: Option<Spline<Interpolator, T>>,
}
impl<
        T: WordAligned
            + Default
            + Interpolate<Interpolator>
            + AddAssign
            + Mul<f64, Output = T>
            + std::cmp::PartialEq
            + std::fmt::Debug,
    > NoiseGenerator<T>
where
    Standard: Distribution<T>,
    [T]: rand::Fill,
{
    /// Constructs a NoiseGenerator with a new RNG source seeded with the seed
    pub fn from_u64_seed(seed: u64, frequency: f64, amplitude: f64, octaves: usize) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            frequency,
            amplitude,
            octaves,
            offset_size: (size_of::<T>() * 8) / 32,
            current_spline: None,
        }
    }
    /// Constructs a NoiseGenerator from an existing RNG source
    pub fn from_rng(rng: ChaCha20Rng, frequency: f64, amplitude: f64, octaves: usize) -> Self {
        Self {
            rng,
            frequency,
            amplitude,
            octaves,
            offset_size: (size_of::<T>() * 8) / 32,
            current_spline: None,
        }
    }
    /// Generates a random number after offsetting the RNG by the offset
    /// This means that asking for an offset of 5 will give the same number
    /// as if generating and discarding 5 numbers and then generating a 6th
    fn random_with_offset(&mut self, i: usize, octave: usize) -> T {
        let mut rng = self.rng.clone();
        rng.set_stream(octave as u64);
        rng.set_word_pos((i * self.offset_size) as u128);
        rng.gen()
    }
    /// Fill the supplied array with noise of length `SIZE` offset by `start_offset`.
    /// This is the same result as calling `sample(x)` with x from `start_offset` to `start_offset + SIZE`
    /// but with the optimisation of the being able to reduce the amount of duplicated calls.
    ///
    /// This optimsation is due to `sample` needing to generate 2 points and the interpolate x between them, for every
    /// invocation. Wheras, as we are garunteed to be generating contiguously, we can generate the minimum amount of points
    /// and resuse them as we interpolate between them.
    ///
    /// For example if you want noise of length 256 with a frequency of 1/32 then `sample` will need to find the current
    /// wave_index (`x / wave_length`) and the next one and then interpolate the value at the wave_position (`x % wave_length`)
    /// for every call, so 256 times, that is 512 random numbers being generated.
    /// However `fill` can see that there will only be `SIZE / wave_length + 1` (9 in this case) random numbers needed
    /// and then reuse them when doing the interpolating step.
    /// This means that there will be 64 times less calls to the RNG for the same result.
    pub fn fill<const SIZE: usize>(&mut self, start_offset: usize, dest: &mut [T; SIZE]) {
        self.rng
            .set_word_pos((start_offset * self.offset_size) as u128);
        let wave_length = (1. / self.frequency) as usize;
        let random_size = (SIZE / wave_length) + 1;
        let mut octaves = self.octaves;
        let amplitude = self.amplitude;
        let mut divisor = 1;
        if octaves > 1 {
            divisor = gcd(wave_length, octaves);
            if divisor == 1 {
                divisor = 2;
                octaves += 1;
            }
        }
        let mut random_numbers = Vec::<Vec<T>>::with_capacity(octaves);
        for octave in 0..octaves {
            let random_size = random_size * divisor.pow((octave+1) as u32);
            let mut octave_random_numbers = vec![T::default(); random_size];
            let mut rng = self.rng.clone();
            rng.set_stream(octave as u64);
            rng.fill(octave_random_numbers.as_mut_slice());
            random_numbers.push(octave_random_numbers);
        }

        for (x, result) in dest.iter_mut().enumerate().take(SIZE) {
            let mut wave_length = wave_length;
            let mut amplitude = amplitude;
            for random_numbers in &random_numbers {
                let wave_position = x % wave_length;
                let wave_index = x / wave_length;
                let y: T = if wave_position == 0 {
                    random_numbers[wave_index]
                } else {
                    let a = random_numbers[wave_index];
                    let b = random_numbers[wave_index + 1];
                    self.interpolate(
                        a,
                        b,
                        (wave_position as Interpolator) / (wave_length as Interpolator),
                    )
                };
                *result += y * amplitude;
                wave_length /= divisor;
                amplitude /= 2.;
            }
        }
    }

    /// Samples the noise at the specific offset, this is an O(1) operation in respect to the offset
    pub fn sample(&mut self, x: usize) -> T {
        let mut wave_length = (1. / self.frequency) as usize;
        let mut result = T::default();
        let mut octaves = self.octaves;
        let mut amplitude = self.amplitude;
        let mut divisor = 1;
        if octaves > 1 {
            divisor = gcd(wave_length, octaves);
            if divisor == 1 {
                divisor = 2;
                octaves += 1;
            }
        }
        for octave in 0..octaves {
            if wave_length == 0 {
                break;
            }
            let wave_position = x % wave_length;
            let wave_index = x / wave_length;
            let y: T = if wave_position == 0 {
                self.random_with_offset(wave_index, octave)
            } else {
                let a = self.random_with_offset(wave_index, octave);
                let b = self.random_with_offset(wave_index + 1, octave);
                self.interpolate(
                    a,
                    b,
                    (wave_position as Interpolator) / (wave_length as Interpolator),
                )
            };
            result += y * amplitude;
            wave_length /= divisor;
            amplitude /= 2.;
        }
        result
    }
    /// Interpolate `x/1` between a and b using a cosine interpolation.
    /// As we are likely going to be interpolating mutiple values between a and b in succession we store the current spline and only replace it when either a or b changes.
    /// This might be a performance improvement and should really have benchmarks of both with and without done.
    fn interpolate(&mut self, a: T, b: T, x: Interpolator) -> T {
        if let Some(current_spline) = &self.current_spline {
            let current_keys = current_spline.keys();
            if !(current_keys[0].value == a && current_keys[1].value == b) {
                self.current_spline = None;
            }
        };
        if self.current_spline.is_none() {
            let start = Key::new(0., a, Interpolation::Cosine);
            let end = Key::new(1., b, Interpolation::default());
            self.current_spline = Some(Spline::from_vec(vec![start, end]));
        }
        if let Some(value) = &self.current_spline.as_mut().unwrap().sample(x) {
            return *value;
        }
        T::default()
    }
}
/// Marks that a type size is aligned with a word boundary (32 bits as used by ChaCha)
pub trait WordAligned {}
impl WordAligned for u32 {}
impl WordAligned for u64 {}
impl WordAligned for u128 {}
impl WordAligned for i32 {}
impl WordAligned for i64 {}
impl WordAligned for i128 {}
impl WordAligned for f32 {}
impl WordAligned for f64 {}
impl WordAligned for char {}
