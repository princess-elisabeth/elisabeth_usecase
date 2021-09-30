use elisabeth::{u4, Encrypter, SystemParameters, LWE};
use ndarray::ArrayD;
use rand::{prelude::SliceRandom, thread_rng};
use std::{collections::HashMap, env, path::Path};
use vm::HomomorphicNeuralNetwork;

mod fhe;
mod fmnist;
mod vm;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_iter: usize = args[1].parse().unwrap();

    // make the encrypter and the transcrypter
    let ((sk, std_dev_lwe), pk) = SystemParameters::n60.generate_fhe_keys();
    let lwe_size = sk.key_size().to_lwe_size();

    let (mut encrypter, mut transcrypter) = Encrypter::<u4>::new::<LWE>(
        &SystemParameters::n60,
        Some(&sk),
        Some(std_dev_lwe.0),
        Some(pk.clone()),
    );

    let model = HomomorphicNeuralNetwork::new(Path::new(
        "data/circuit/fmnist_zonnx_N_2048/elisabeth_fmnist_exe",
    ));

    // get samples
    let mut images = fmnist::load_data("t10k", Some(3)).unwrap();
    let mut rng = thread_rng();
    images.shuffle(&mut rng);
    let images = &images[..n_iter];

    let mut errors = 0.;
    let start_time = std::time::Instant::now();
    for (i, image) in images.into_iter().enumerate() {
        let mut res = vec![u4(0); image.pixels.len()];
        encrypter.encrypt(&mut res, image.pixels.as_slice().unwrap());

        let mut sample_shape = vec![1];
        sample_shape.append(&mut image.pixels.shape().to_vec());
        let encrypted_sample =
            ArrayD::from_shape_vec(sample_shape, res.iter().map(|u| u.0).collect()).unwrap();

        let inputs = std::iter::once((
            "0".to_string(),
            fhe::transcrypt(&mut transcrypter, &encrypted_sample, lwe_size),
        ))
        .collect::<HashMap<_, _>>();

        let outputs = model.run(inputs, &pk);

        let decoded = &outputs
            .into_iter()
            .map(|a| fhe::decrypt(&sk, &a.into_dyn().view()))
            .collect::<Vec<_>>()[0];

        let mut label_weights = decoded.iter().enumerate().collect::<Vec<_>>();

        label_weights.sort_by(|(_label1, weight1), (_label2, weight2)| {
            weight1
                .partial_cmp(weight2)
                .expect("Uncomparable element found (could be Nan)")
        });
        if label_weights.last().unwrap().0 != image.classification as usize {
            errors += 1.;
        }
        println!(
            "📈 Accuracy: {}% over {} input{}.",
            100. - 100. * errors / (i as f64 + 1.),
            i + 1,
            if i > 0 { "s" } else { "" }
        );
    }

    println!(
        "Total execution time for {} samples: {} s",
        n_iter,
        start_time.elapsed().as_secs_f64()
    );
}
