# Usecase
This is a Rust implementation of a usecase for the *Elisabeth* stream cipher. this runs homomorphic inferences on transciphered data from the F-MNIST dataset.

## Prerequisite

To run this project, you will need the Rust compiler, and the FFTW library. The compiler can be
installed on linux and osx with the following command:

```bash
curl  --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Other rust installation methods are available on the
[rust website](https://forge.rust-lang.org/infra/other-installation-methods.html).

To install the FFTW library on MacOS, one could use the Homebrew package manager. To install
Homebrew, you can do the following:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

And then use it to install FFTW:

```bash
brew install fftw
```

To install FFTW on a debian-based distribution, you can use the following command:

```bash
sudo apt-get update && sudo apt-get install -y libfftw3-dev
```

You can then clone this repository by doing:

```bash
git clone git@github.com:princess-elisabeth/elisabeth_usecase.git
```

Moreover, this project requires access to the closed beta of Zama's Virtual Machine. To gain access to this beta, please contact hello@zama.ai and ask to be added to both `zamavm-beta` and `concrete_internal-beta` repositories. 

## Usage
Before running any test or benchmark, you should export the following RUSTFLAGS:
```
export RUSTFLAGS="-C target-cpu=native"
```

Then run the project with the following command:
```bash
cargo run --release -- *NUMBER_OF_INFERENCES*
```
Where *NUMBER_OF_INFERENCES* should be replaced by the actual number of inferences you want to run.

Note that an inference should be several minutes long.