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
#[derive(Clone,Debug)]
pub struct NoiseGenerator<
    T: WordAligned + Default + Interpolate<Interpolator> + AddAssign + Mul<f64, Output = T>,
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
            + std::cmp::PartialEq,
    > NoiseGenerator<T>
where
    Standard: Distribution<T>,
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
    fn random_with_offset(&mut self, i: usize) -> T {
        self.rng.set_word_pos((i * self.offset_size) as u128);
        self.rng.gen()
    }
    /// Samples the noise at the specific offset, this is an O(1) operation in respect to the offset
    pub fn sample(&mut self, x: usize) -> T {
        let mut wave_length = (1. / self.frequency) as usize;
        let mut result = T::default();
        let mut octaves = self.octaves;
        let mut amplitude = self.amplitude;
        let mut divisor = gcd(wave_length, octaves);
        if divisor == 1 {
            divisor = 2;
            octaves += 1;
        }
        for _ in 0..octaves {
            if wave_length == 0 {break;}
            let wave_position = x % wave_length;
            let wave_index = x / wave_length;
            let y: T = if wave_position == 0 {
                self.random_with_offset(wave_index)
            } else {
                let a = self.random_with_offset(wave_index);
                let b = self.random_with_offset(wave_index + 1);
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
