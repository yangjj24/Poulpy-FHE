//! Hoisted base-B mux blind rotation (SHIP §5.1, Algorithm 5).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    default::keyswitching::glwe::{GGLWEProductDefault, gglwe_product_accumulation_output_size},
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GGLWEPreparedToBackendRef},
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    },
};

use std::collections::HashMap;

use crate::layouts::{CKKSCiphertextOwned, ship::keyset::HMuxRotKeyPrepared};

/// Automorphism plans of the mux output twists, keyed by Galois element.
pub(crate) type ShipMuxPlans<BE> = HashMap<i64, <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan>;

/// Builds the automorphism plans of every distinct mux Galois element once;
/// the same rotation amounts recur across all support slots.
pub(crate) fn ship_mux_plans<'a, BE>(
    module: &Module<BE>,
    groups: impl Iterator<Item = &'a [HMuxRotKeyPrepared<BE::OwnedBuf, BE>]>,
) -> ShipMuxPlans<BE>
where
    BE: Backend + 'a,
    Module<BE>: VecZnxDftAutomorphismPlan<BE>,
{
    let mut plans = ShipMuxPlans::<BE>::new();
    for group in groups {
        for key in group {
            if key.gal_el != 1 {
                plans
                    .entry(key.gal_el)
                    .or_insert_with(|| module.vec_znx_dft_automorphism_plan(key.gal_el));
            }
        }
    }
    plans
}

/// Scratch bytes for [`ship_mux_rotate`].
pub(crate) fn ship_mux_rotate_tmp_bytes<BE, C, K>(module: &Module<BE>, ct: &C, key: &K, term_count: usize) -> usize
where
    BE: Backend,
    C: LWEInfos,
    K: GGLWEInfos,
    Module<BE>: VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + GGLWEProductDefault<BE>,
{
    let a_size = ct.size();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(ct, ct, key, term_count);

    let product = module.gglwe_product_dft_tmp_bytes_default(output_size, a_size, key);
    let mux = 2 * module.bytes_of_vec_znx_dft(2, output_size) + product;
    let finalize = module.bytes_of_vec_znx_big(2, output_size) + module.vec_znx_big_normalize_tmp_bytes();
    let single = module.bytes_of_vec_znx_dft(2, a_size) + module.bytes_of_vec_znx_dft(2, output_size) + mux.max(finalize);

    // single.max(ship_mux_rotate_dual_tmp_bytes(module, ct, key, term_count))
    single
}

/// Scratch bytes for [`ship_mux_rotate_dual`]. Two input DFTs and two sums are
/// live together; per-key product/automorphism temporaries are also paired.
pub(crate) fn ship_mux_rotate_dual_tmp_bytes<BE, C, K>(module: &Module<BE>, ct: &C, key: &K, term_count: usize) -> usize
where
    BE: Backend,
    C: LWEInfos,
    K: GGLWEInfos,
    Module<BE>: VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + GGLWEProductDefault<BE>,
{
    let a_size = ct.size();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(ct, ct, key, term_count);
    let product = module.gglwe_product_dft_dual_tmp_bytes_default(output_size, a_size, key);
    let pair = module.bytes_of_vec_znx_dft(2, output_size);
    let mux = 4 * pair + product;
    let finalize = module.bytes_of_vec_znx_big(2, output_size) + module.vec_znx_big_normalize_tmp_bytes();
    2 * module.bytes_of_vec_znx_dft(2, a_size) + 2 * pair + mux.max(finalize)
}

/// Hoisted B-to-1 mux-rotate: `ct <- sum_d beta_d * Rot_{rot_d}(ct)` over the
/// keys of one digit position. The rank-2 input `(0, a_mask, a_body)` is
/// DFT'd once and shared by every key; each key contributes one VMP followed
/// by its DFT-domain output automorphism, and a single IDFT + normalize
/// closes the position.
pub(crate) fn ship_mux_rotate<BE>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextOwned<BE>,
    keys: &[HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
    plans: &ShipMuxPlans<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + GGLWEProductDefault<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_mux_rotate";
    ckks_ensure!(!keys.is_empty(), "{OP}: empty key group");
    let a_size = ct.size();
    let key_size = keys[0].key.size();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(ct, ct, &keys[0].key, keys.len());
    let base2k = ct.base2k().as_usize();
    ckks_ensure!(
        keys[0].key.base2k().as_usize() == base2k,
        "{OP}: ciphertext/key base2k mismatch"
    );

    let scratch = scratch.borrow();
    let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 2, a_size);
    {
        let a_ref = GLWEToBackendRef::<BE>::to_backend_ref(ct);
        let mut a_dft_mut = a_dft.to_backend_mut();
        module.vec_znx_dft_apply(1, 0, &mut a_dft_mut, 0, a_ref.data(), 1);
        module.vec_znx_dft_apply(1, 0, &mut a_dft_mut, 1, a_ref.data(), 0);
    }
    let a_dft_ref = a_dft.to_backend_ref();

    let (mut sum_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, 2, output_size);
    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_dft_zero(&mut sum_dft_mut, col);
        }
        let (mut prod_dft, scratch_3) = scratch_2.borrow().take_vec_znx_dft_scratch(module, 2, output_size);
        let (mut rot_dft, mut scratch_4) = scratch_3.take_vec_znx_dft_scratch(module, 2, output_size);
        for key in keys {
            ckks_ensure!(key.key.size() == key_size, "{OP}: inconsistent key sizes in group");
            {
                let mut prod_dft_mut = prod_dft.to_backend_mut();
                module.gglwe_product_dft_default(
                    &mut prod_dft_mut,
                    &a_dft_ref,
                    &key.key.to_backend_ref(),
                    keys.len(),
                    &mut scratch_4.borrow(),
                );
            }
            if key.gal_el == 1 {
                let prod_ref = prod_dft.to_backend_ref();
                for col in 0..2 {
                    module.vec_znx_dft_add_assign(&mut sum_dft_mut, col, &prod_ref, col);
                }
            } else {
                let plan = plans
                    .get(&key.gal_el)
                    .ok_or_else(|| anyhow::anyhow!("{OP}: missing automorphism plan for Galois element {}", key.gal_el))?;
                {
                    let prod_ref = prod_dft.to_backend_ref();
                    let mut rot_dft_mut = rot_dft.to_backend_mut();
                    for col in 0..2 {
                        module.vec_znx_dft_automorphism_with_plan(plan, &mut rot_dft_mut, col, &prod_ref, col);
                    }
                }
                let rot_ref = rot_dft.to_backend_ref();
                for col in 0..2 {
                    module.vec_znx_dft_add_assign(&mut sum_dft_mut, col, &rot_ref, col);
                }
            }
        }
    }

    let (mut res_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(module, 2, output_size);
    {
        let mut res_big_mut = res_big.to_backend_mut();
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_dft_mut, col);
        }
    }
    let k = ct.k().as_usize();
    {
        let res_big_ref = res_big.to_backend_ref();
        let mut ct_mut = ct.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                ct_mut.data_mut(),
                base2k,
                k,
                0,
                col,
                &res_big_ref,
                base2k,
                col,
                &mut scratch_3.borrow(),
            );
        }
    }
    Ok(())
}

/// Dual hoisted B-to-1 mux-rotate for the two complex SHIP coefficient halves.
/// The ciphertext arithmetic remains independent, while each prepared H-MUX
/// key is traversed once at the GGLWE/VMP level for both inputs.
pub(crate) fn ship_mux_rotate_dual<BE>(
    module: &Module<BE>,
    ct0: &mut CKKSCiphertextOwned<BE>,
    ct1: &mut CKKSCiphertextOwned<BE>,
    keys: &[HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
    plans: &ShipMuxPlans<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + GGLWEProductDefault<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_mux_rotate_dual";
    ckks_ensure!(!keys.is_empty(), "{OP}: empty key group");
    ckks_ensure!(ct0.size() == ct1.size(), "{OP}: ciphertext sizes differ");
    ckks_ensure!(ct0.base2k() == ct1.base2k(), "{OP}: ciphertext base2k differs");
    ckks_ensure!(ct0.k() == ct1.k(), "{OP}: ciphertext precisions differ");

    let a_size = ct0.size();
    let key_size = keys[0].key.size();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(ct0, ct0, &keys[0].key, keys.len());
    let base2k = ct0.base2k().as_usize();
    ckks_ensure!(
        keys[0].key.base2k().as_usize() == base2k,
        "{OP}: ciphertext/key base2k mismatch"
    );

    let scratch = scratch.borrow();
    let (mut a0_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 2, a_size);
    let (mut a1_dft, scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, 2, a_size);
    {
        let a0_ref = GLWEToBackendRef::<BE>::to_backend_ref(ct0);
        let a1_ref = GLWEToBackendRef::<BE>::to_backend_ref(ct1);
        let mut a0_mut = a0_dft.to_backend_mut();
        let mut a1_mut = a1_dft.to_backend_mut();
        module.vec_znx_dft_apply(1, 0, &mut a0_mut, 0, a0_ref.data(), 1);
        module.vec_znx_dft_apply(1, 0, &mut a0_mut, 1, a0_ref.data(), 0);
        module.vec_znx_dft_apply(1, 0, &mut a1_mut, 0, a1_ref.data(), 1);
        module.vec_znx_dft_apply(1, 0, &mut a1_mut, 1, a1_ref.data(), 0);
    }
    let a0_ref = a0_dft.to_backend_ref();
    let a1_ref = a1_dft.to_backend_ref();

    let (mut sum0, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module, 2, output_size);
    let (mut sum1, mut work) = scratch_3.take_vec_znx_dft_scratch(module, 2, output_size);
    {
        let mut sum0_mut = sum0.to_backend_mut();
        let mut sum1_mut = sum1.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_dft_zero(&mut sum0_mut, col);
            module.vec_znx_dft_zero(&mut sum1_mut, col);
        }

        let (mut prod0, work_1) = work.borrow().take_vec_znx_dft_scratch(module, 2, output_size);
        let (mut prod1, work_2) = work_1.take_vec_znx_dft_scratch(module, 2, output_size);
        let (mut rot0, work_3) = work_2.take_vec_znx_dft_scratch(module, 2, output_size);
        let (mut rot1, mut product_scratch) = work_3.take_vec_znx_dft_scratch(module, 2, output_size);

        for key in keys {
            ckks_ensure!(key.key.size() == key_size, "{OP}: inconsistent key sizes in group");
            {
                let mut prod0_mut = prod0.to_backend_mut();
                let mut prod1_mut = prod1.to_backend_mut();
                module.gglwe_product_dft_dual_default(
                    &mut prod0_mut,
                    &mut prod1_mut,
                    &a0_ref,
                    &a1_ref,
                    &key.key.to_backend_ref(),
                    keys.len(),
                    &mut product_scratch.borrow(),
                );
            }

            if key.gal_el == 1 {
                let prod0_ref = prod0.to_backend_ref();
                let prod1_ref = prod1.to_backend_ref();
                for col in 0..2 {
                    module.vec_znx_dft_add_assign(&mut sum0_mut, col, &prod0_ref, col);
                    module.vec_znx_dft_add_assign(&mut sum1_mut, col, &prod1_ref, col);
                }
            } else {
                let plan = plans
                    .get(&key.gal_el)
                    .ok_or_else(|| anyhow::anyhow!("{OP}: missing automorphism plan for Galois element {}", key.gal_el))?;
                {
                    let prod0_ref = prod0.to_backend_ref();
                    let prod1_ref = prod1.to_backend_ref();
                    let mut rot0_mut = rot0.to_backend_mut();
                    let mut rot1_mut = rot1.to_backend_mut();
                    for col in 0..2 {
                        module.vec_znx_dft_automorphism_with_plan(plan, &mut rot0_mut, col, &prod0_ref, col);
                        module.vec_znx_dft_automorphism_with_plan(plan, &mut rot1_mut, col, &prod1_ref, col);
                    }
                }
                let rot0_ref = rot0.to_backend_ref();
                let rot1_ref = rot1.to_backend_ref();
                for col in 0..2 {
                    module.vec_znx_dft_add_assign(&mut sum0_mut, col, &rot0_ref, col);
                    module.vec_znx_dft_add_assign(&mut sum1_mut, col, &rot1_ref, col);
                }
            }
        }
    }

    let k = ct0.k().as_usize();
    for half in 0..2 {
        let (mut res_big, mut normalize_scratch) = work.borrow().take_vec_znx_big_scratch(module, 2, output_size);
        {
            let mut res_big_mut = res_big.to_backend_mut();
            if half == 0 {
                let mut sum_mut = sum0.to_backend_mut();
                for col in 0..2 {
                    module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_mut, col);
                }
            } else {
                let mut sum_mut = sum1.to_backend_mut();
                for col in 0..2 {
                    module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_mut, col);
                }
            }
        }
        let res_big_ref = res_big.to_backend_ref();
        let ct = if half == 0 { &mut *ct0 } else { &mut *ct1 };
        let mut ct_mut = ct.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                ct_mut.data_mut(),
                base2k,
                k,
                0,
                col,
                &res_big_ref,
                base2k,
                col,
                &mut normalize_scratch.borrow(),
            );
        }
    }
    Ok(())
}
