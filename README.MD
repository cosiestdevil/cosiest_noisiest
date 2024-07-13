[![Crates.io Version](https://img.shields.io/crates/v/cosiest_noisiest)](https://crates.io/crates/cosiest_noisiest)
[![DeepSource](https://app.deepsource.com/gh/cosiestdevil/cosiest_noisiest.svg/?label=active+issues&show_trend=true&token=OAOF2YZWASVVnyP4i6mdMFRy)](https://app.deepsource.com/gh/cosiestdevil/cosiest_noisiest/)

# Cosiest Noisiest

`cosiest_noisiest` allows generating 1d noise at arbitrary points

# Usage
Add this to your `Cargo.toml`
```toml
[dependencies]
cosiest_noisiest = "0.1.1"
```

# Examples
```rust
use cosiest_noisiest::NoiseGenerator;
let mut noise_generator = NoiseGenerator::from_u64_seed(2, 1. / 32., 128., 3);
let noise:f64 = noise_generator.sample(1024);
```
```rust
use cosiest_noisiest::NoiseGenerator;
let mut noise_generator = NoiseGenerator::from_u64_seed(2, 1. / 32., 128., 3);
let noise = [0.0;1024];
noise_generator.fill(noise);
```

# Crate Features
 **f32** - Enables using f32 when interpolating between noise values instead of the default f64.
