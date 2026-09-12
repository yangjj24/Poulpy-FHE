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

/// Hoisted B-to-1 mux-rotate with an explicit bit offset at the final
/// normalization. Ordinary SHIP uses offset zero; packed selectors use a
/// negative offset to remove their fixed-point encoding scale.
pub(crate) fn ship_mux_rotate_with_offset<BE>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextOwned<BE>,
    keys: &[HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
    plans: &ShipMuxPlans<BE>,
    res_offset: i64,
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
                res_offset,
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

/// Fused multi-source H-MUX with an explicit final normalization offset.
///
/// `sources[j]` is paired with `keys[j]`.  Unlike [`ship_mux_rotate_with_offset`],
/// which hoists one source ciphertext across several selector keys, this routine
/// accepts a different source ciphertext for every selector/rotation term:
///
/// `out <- sum_j HMuxRot(keys[j], sources[j])`.
///
/// The source DFT buffer and product/rotation temporaries are reused term by
/// term; only the DFT-domain accumulator remains live across all terms.  Thus
/// all products share one final IDFT + normalization.
pub(crate) fn ship_mux_rotate_multi_source_refs_with_offset<BE>(
    module: &Module<BE>,
    out: &mut CKKSCiphertextOwned<BE>,
    sources: &[&CKKSCiphertextOwned<BE>],
    keys: &[&HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
    plans: &ShipMuxPlans<BE>,
    res_offset: i64,
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
    const OP: &str = "ship_mux_rotate_multi_source_refs";
    ckks_ensure!(!sources.is_empty(), "{OP}: empty source list");
    ckks_ensure!(
        sources.len() == keys.len(),
        "{OP}: source/key count mismatch ({} vs {})",
        sources.len(),
        keys.len()
    );

    let first = sources[0];
    let a_size = first.size();
    let key_size = keys[0].key.size();
    let base2k = first.base2k().as_usize();
    let k = first.k().as_usize();

    ckks_ensure!(
        out.base2k().as_usize() == base2k && out.k().as_usize() == k,
        "{OP}: output/source precision mismatch"
    );

    for (source, key) in sources.iter().zip(keys.iter()) {
        ckks_ensure!(source.size() == a_size, "{OP}: inconsistent source sizes");
        ckks_ensure!(
            source.base2k().as_usize() == base2k && source.k().as_usize() == k,
            "{OP}: inconsistent source precision"
        );
        ckks_ensure!(key.key.size() == key_size, "{OP}: inconsistent key sizes");
        ckks_ensure!(key.key.base2k().as_usize() == base2k, "{OP}: ciphertext/key base2k mismatch");
    }

    // `keys.len()` is the total number of products that share this accumulator;
    // exact backends use it to select the safe shortened accumulation width.
    let term_count = keys.len();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(first, first, &keys[0].key, term_count);

    let scratch = scratch.borrow();
    let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 2, a_size);
    let (mut sum_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, 2, output_size);

    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_dft_zero(&mut sum_dft_mut, col);
        }

        let (mut prod_dft, scratch_3) = scratch_2.borrow().take_vec_znx_dft_scratch(module, 2, output_size);
        let (mut rot_dft, mut scratch_4) = scratch_3.take_vec_znx_dft_scratch(module, 2, output_size);

        for (source, key) in sources.iter().zip(keys.iter()) {
            {
                let source_ref = GLWEToBackendRef::<BE>::to_backend_ref(*source);
                let mut a_dft_mut = a_dft.to_backend_mut();
                module.vec_znx_dft_apply(1, 0, &mut a_dft_mut, 0, source_ref.data(), 1);
                module.vec_znx_dft_apply(1, 0, &mut a_dft_mut, 1, source_ref.data(), 0);
            }

            {
                let a_dft_ref = a_dft.to_backend_ref();
                let mut prod_dft_mut = prod_dft.to_backend_mut();
                module.gglwe_product_dft_default(
                    &mut prod_dft_mut,
                    &a_dft_ref,
                    &key.key.to_backend_ref(),
                    term_count,
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

    {
        let res_big_ref = res_big.to_backend_ref();
        let mut out_mut = out.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                out_mut.data_mut(),
                base2k,
                k,
                res_offset,
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

/// Compatibility wrapper for callers that own a contiguous key group.
///
/// The borrowed-key variant above is the actual implementation. Keeping this
/// wrapper preserves the Phase-3C/4A call sites while allowing Phase 4C and the
/// eventual SHIP runtime to reuse one prepared key in several output groups.
pub(crate) fn ship_mux_rotate_multi_source_with_offset<BE>(
    module: &Module<BE>,
    out: &mut CKKSCiphertextOwned<BE>,
    sources: &[&CKKSCiphertextOwned<BE>],
    keys: &[HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
    plans: &ShipMuxPlans<BE>,
    res_offset: i64,
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
    let key_refs: Vec<_> = keys.iter().collect();
    ship_mux_rotate_multi_source_refs_with_offset(module, out, sources, &key_refs, plans, res_offset, scratch)
}

/// Ordinary scalar-selector H-MUX: no extra fixed-point scale.
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
    ship_mux_rotate_with_offset(module, ct, keys, plans, 0, scratch)
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

/// Logical-to-physical routing for the experimental sheared SHIP layout.
///
/// Layout invariant (slot domain):
///     Y_g[p] = x_{p mod G}[(p + g) mod n].
///
/// A common logical rotation by `t` is obtained from one source group and a
/// physical rotation that is always a multiple of `G`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShipShearedRoute {
    pub(crate) group: usize,
    pub(crate) physical_rot: usize,
}

#[inline]
pub(crate) fn ship_sheared_branch(p: usize, groups: usize) -> usize {
    debug_assert!(groups != 0);
    p % groups
}

#[inline]
pub(crate) fn ship_sheared_logical_index(p: usize, group: usize, slots: usize) -> usize {
    debug_assert!(slots != 0);
    (p + group) % slots
}

/// Route an output group and a non-negative logical rotation to its source.
#[inline]
pub(crate) fn ship_sheared_route_from_output(out: usize, logical_rot: usize, groups: usize) -> ShipShearedRoute {
    debug_assert!(groups != 0 && out < groups);
    let src = (out + logical_rot) % groups;
    let physical_rot = out + logical_rot - src;
    debug_assert_eq!(physical_rot % groups, 0);
    ShipShearedRoute {
        group: src,
        physical_rot,
    }
}

/// Source-major form of [`ship_sheared_route_from_output`].
#[inline]
pub(crate) fn ship_sheared_route_from_source(src: usize, logical_rot: usize, groups: usize) -> ShipShearedRoute {
    debug_assert!(groups != 0 && src < groups);
    let out = (src + groups - (logical_rot % groups)) % groups;
    let physical_rot = out + logical_rot - src;
    debug_assert_eq!(physical_rot % groups, 0);
    ShipShearedRoute {
        group: out,
        physical_rot,
    }
}

#[cfg(test)]
mod sheared_layout_tests {
    use super::ship_sheared_runtime_route_from_output;

    use super::{
        ship_sheared_branch, ship_sheared_logical_index, ship_sheared_route_from_output, ship_sheared_route_from_source,
    };

    #[test]
    fn sheared_layout_keeps_branch_in_fixed_residue_lane() {
        for groups in [2usize, 4, 8, 32] {
            let slots = groups * 8;
            for g in 0..groups {
                for p in 0..slots {
                    assert_eq!(ship_sheared_branch(p, groups), p % groups);
                    assert_eq!(ship_sheared_logical_index(p, g, slots), (p + g) % slots);
                }
            }
        }
    }

    #[test]
    fn sheared_route_from_output_matches_logical_rotation() {
        for groups in [2usize, 4, 8, 32] {
            let slots = groups * 8;
            for out in 0..groups {
                for logical_rot in 0..slots {
                    let route = ship_sheared_route_from_output(out, logical_rot, groups);
                    assert_eq!(route.physical_rot % groups, 0);
                    for p in 0..slots {
                        let routed_index = (p + route.physical_rot + route.group) % slots;
                        let expected_index = (p + out + logical_rot) % slots;
                        assert_eq!(routed_index, expected_index);
                    }
                }
            }
        }
    }

    #[test]
    fn sheared_source_and_output_routes_are_inverses() {
        for groups in [2usize, 4, 8, 32] {
            let slots = groups * 8;
            for src in 0..groups {
                for logical_rot in 0..slots {
                    let forward = ship_sheared_route_from_source(src, logical_rot, groups);
                    let backward = ship_sheared_route_from_output(forward.group, logical_rot, groups);
                    assert_eq!(backward.group, src);
                    assert_eq!(backward.physical_rot, forward.physical_rot);
                }
            }
        }
    }

    #[test]
    fn sheared_runtime_route_supports_g64_padding() {
        const GROUPS: usize = 64;
        const SLOTS: usize = 128;

        for runtime_rot in [0usize, 4, 8, 12, 16, 32, 48, 64] {
            for out in 0..GROUPS {
                let route = ship_sheared_runtime_route_from_output(out, runtime_rot, GROUPS);

                assert_eq!(route.physical_rot % GROUPS, 0);
                assert_eq!((route.group + runtime_rot) % GROUPS, out);

                for p in 0..SLOTS {
                    let physical_p = (p + SLOTS - (route.physical_rot % SLOTS)) % SLOTS;
                    let have = (physical_p + route.group) % SLOTS;
                    let want = (p + out + SLOTS - (runtime_rot % SLOTS)) % SLOTS;

                    assert_eq!(
                        have, want,
                        "G=64 route mismatch: rot={runtime_rot}, out={out}, p={p}, route={route:?}"
                    );
                    assert_eq!(
                        physical_p % GROUPS,
                        p % GROUPS,
                        "G=64 physical rotation changed the fixed residue lane"
                    );
                }
            }
        }
    }
}

/// Maps one positive Poulpy/H-MUX rotation to the sheared layout.
///
/// Poulpy's runtime convention is `Rot_t(x)[i] = x[i - t]`. For
///
/// `Y_g[p] = x_{p mod G}[p + g]`,
///
/// output group `g` therefore consumes source group
/// `s = (g - t) mod G` and only needs the physical rotation
///
/// `rho = s + t - g`,
///
/// which is always a non-negative multiple of `G`. Because `rho mod G = 0`,
/// the fixed branch lane `p mod G` is preserved by the physical rotation.
#[inline]
pub(crate) fn ship_sheared_runtime_route_from_output(out: usize, runtime_rot: usize, groups: usize) -> ShipShearedRoute {
    debug_assert!(groups != 0 && out < groups);
    let t = runtime_rot % groups;
    let src = (out + groups - t) % groups;
    let physical_rot = src + runtime_rot - out;
    debug_assert_eq!(physical_rot % groups, 0);
    ShipShearedRoute {
        group: src,
        physical_rot,
    }
}

#[cfg(test)]
mod sheared_runtime_route_tests {
    use super::{ship_sheared_branch, ship_sheared_logical_index, ship_sheared_runtime_route_from_output};

    #[test]
    fn runtime_route_matches_hmux_minus_rotation_convention() {
        const GROUPS: usize = 32;
        const SLOTS: usize = GROUPS * 8;

        for out in 0..GROUPS {
            for runtime_rot in 0..SLOTS {
                let route = ship_sheared_runtime_route_from_output(out, runtime_rot, GROUPS);

                assert_eq!(route.physical_rot % GROUPS, 0);
                assert_eq!((route.group + runtime_rot) % GROUPS, out);

                for p in 0..SLOTS {
                    // Poulpy/H-MUX: positive rotation t reads input[p - t].
                    let in_p = (p + SLOTS - (route.physical_rot % SLOTS)) % SLOTS;

                    assert_eq!(ship_sheared_branch(in_p, GROUPS), ship_sheared_branch(p, GROUPS),);

                    // Y_src[p-rho] = x_lane[p + out - runtime_rot].
                    let got = ship_sheared_logical_index(in_p, route.group, SLOTS);
                    let want = (p + out + SLOTS - (runtime_rot % SLOTS)) % SLOTS;
                    assert_eq!(got, want);
                }
            }
        }
    }
}
