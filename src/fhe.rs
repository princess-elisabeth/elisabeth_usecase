use concrete_core::{
    crypto::{encoding::Plaintext, lwe::LweCiphertext, secret::LweSecretKey, LweSize},
    math::tensor::{AsRefSlice, AsRefTensor},
};
use elisabeth::{u4, Encrypter, LWE};
use ndarray::{Array, ArrayD, ArrayView, Axis, Dimension, Zip};

pub fn transcrypt(
    transcrypter: &mut Encrypter<LWE>,
    value: &ArrayD<u8>,
    lwe_size: LweSize,
) -> ArrayD<u64> {
    let mut out_shape = value.shape().to_vec();
    out_shape.push(lwe_size.0);

    let mut result = ArrayD::zeros(out_shape.as_slice())
        .into_dimensionality()
        .unwrap();

    let mut res = vec![LWE::allocate(lwe_size); value.len()];
    transcrypter.decrypt(&mut res, &value.iter().map(|u| u4(*u)).collect::<Vec<_>>());
    result
        .as_slice_mut()
        .unwrap()
        .chunks_mut(lwe_size.0)
        .zip(res.iter())
        .for_each(|(slice, lwe)| slice.clone_from_slice(lwe.as_lwe().as_tensor().as_slice()));
    result
}

pub fn decrypt<D: Dimension>(
    sk: &LweSecretKey<Vec<bool>>,
    tensor: &ArrayView<u64, D>,
) -> Array<u64, D::Smaller> {
    let last_dimension = tensor.shape().len() - 1;

    let (_, out_shape) = tensor.shape().split_last().unwrap();

    let mut result: Array<u64, D::Smaller> =
        ArrayD::zeros(out_shape).into_dimensionality().unwrap();

    Zip::from(result.view_mut())
        .and(tensor.lanes(Axis(last_dimension)))
        .for_each(|value, ciphertext| {
            let mut output = Plaintext(0);
            sk.decrypt_lwe(
                &mut output,
                &LweCiphertext::from_container(ciphertext.as_slice().unwrap()),
            );

            *value = output.0;
        });

    result
}
