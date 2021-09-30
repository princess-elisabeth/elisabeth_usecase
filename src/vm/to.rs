use concrete_core::{
    crypto::{bootstrap::BootstrapKey, lwe::LweKeyswitchKey},
    math::{
        fft::Complex64,
        tensor::{AsMutSlice, AsRefSlice, AsRefTensor},
    },
};
use crypto::keys::{CloudKeys, KsKey, PbsKey};
use elisabeth::PublicKey;

pub(super) trait ToVM {
    type VMType;
    fn to_vm(&self) -> Self::VMType;
}

impl ToVM for BootstrapKey<Vec<Complex64>> {
    type VMType = PbsKey;

    fn to_vm(&self) -> Self::VMType {
        let glwe_size = concrete_vm::crypto::GlweSize(self.glwe_size().0);
        let poly_size = concrete_vm::math::polynomial::PolynomialSize(self.polynomial_size().0);
        let decomp_level =
            concrete_vm::math::decomposition::DecompositionLevelCount(self.level_count().0);
        let base_log = concrete_vm::math::decomposition::DecompositionBaseLog(self.base_log().0);

        let mut aligned_vec =
            concrete_vm::math::fft::AlignedVec::new(self.as_tensor().as_container().len());
        aligned_vec
            .as_mut_slice()
            .clone_from_slice(self.as_tensor().as_slice());

        PbsKey(
            concrete_vm::crypto::bootstrap::BootstrapKey::from_container(
                aligned_vec,
                glwe_size,
                poly_size,
                decomp_level,
                base_log,
            ),
        )
    }
}

impl ToVM for LweKeyswitchKey<Vec<u64>> {
    type VMType = KsKey<u64>;

    fn to_vm(&self) -> Self::VMType {
        let level = self.decomposition_levels_count().0;
        let base_log = self.decomposition_base_log().0;

        KsKey(concrete_vm::crypto::lwe::LweKeyswitchKey::from_container(
            self.as_tensor().as_container().clone(),
            concrete_vm::math::decomposition::DecompositionBaseLog(base_log),
            concrete_vm::math::decomposition::DecompositionLevelCount(level),
            concrete_vm::crypto::LweDimension(self.after_key_size().0),
        ))
    }
}

impl ToVM for PublicKey {
    type VMType = CloudKeys<u64>;

    fn to_vm(&self) -> Self::VMType {
        let pbs_keys = std::iter::once(("bsk".to_string(), self.bsk.to_vm())).collect();
        let ks_keys = std::iter::once(("ksk".to_string(), self.ksk.to_vm())).collect();

        CloudKeys { pbs_keys, ks_keys }
    }
}
