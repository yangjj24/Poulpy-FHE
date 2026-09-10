//! Theta-column masking accumulation (SHIP §3.3 masks fused with the
//! low-digit column selection of the §4.4 hybrid).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
    msb_mask_bottom_limb,
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, CnvDftAccTerm, CnvPVecL, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Module, ScratchArena,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut,
    },
};

use crate::{
    CKKSInfos, SetCKKSInfos,
    default::mul::mul_pt_params_raw,
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, ShipPlan},
};

/// Scratch bytes for [`ship_masking_accumulate`].
pub(crate) fn ship_masking_tmp_bytes<BE>(module: &Module<BE>, plan: &ShipPlan, base2k: usize) -> usize
where
    BE: Backend,
    Module<BE>: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
{
    let a_size = plan.raised_k(base2k).div_ceil(base2k);
    let b_size = (plan.log_delta_work() + base2k).div_ceil(base2k);
    let res_dft_size = a_size + b_size;
    let preps = 4 * plan.theta() * module.bytes_of_cnv_pvec_right(1, b_size);
    let work = module
        .cnv_prepare_right_tmp_bytes(b_size, b_size)
        .max(module.cnv_accumulate_dft_tmp_bytes(0, res_dft_size, a_size, b_size))
        .max(module.cnv_accumulate_dft_dual_tmp_bytes(0, res_dft_size, a_size, b_size))
        .max(module.bytes_of_vec_znx_big(2, res_dft_size) + module.vec_znx_big_normalize_tmp_bytes());
    // The complex path keeps both two-column DFT sums live together.
    // The real-only path over-allocates this temporary by two columns.
    module.bytes_of_vec_znx_dft(4, res_dft_size) + preps + work
}

/// Lazy masking accumulation: `acc = sum_i masks[i] * pis[i]` over the
/// keygen-prepared mask operands, the per-pair convolutions accumulated in
/// the DFT domain with a single IDFT + normalize per column.
pub(crate) fn ship_masking_accumulate<BE>(
    module: &Module<BE>,
    acc: &mut CKKSCiphertextOwned<BE>,
    plan: &ShipPlan,
    masks: &[CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>],
    pis: &[CKKSPlaintextOwned<BE>],
    band_order: [usize; 4],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_masking_accumulate";
    ckks_ensure!(
        !masks.is_empty() && masks.len() == pis.len() && masks.len() % 4 == 0,
        // "{OP}: empty or mismatched operands"
        "{OP}: empty, mismatched, or malformed operands"
    );
    let base2k = acc.base2k().as_usize();
    let kk = plan.raised_k(base2k);
    let ld = plan.log_delta_work();
    let (res_log_budget, res_log_delta, cnv_offset) = mul_pt_params_raw(
        acc.max_k().as_usize(),
        ld,
        kk - ld,
        pis[0].log_delta(),
        pis[0].log_budget(),
        pis[0].max_k().as_usize(),
    )?;
    let a_size = kk.div_ceil(base2k);
    let b_size = pis[0].size();
    let b_mask = msb_mask_bottom_limb(base2k, pis[0].max_k().as_usize());
    let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < base2k {
        (0, -((base2k - (cnv_offset % base2k)) as i64))
    } else {
        ((cnv_offset / base2k).saturating_sub(1), (cnv_offset % base2k) as i64)
    };
    let res_dft_size = a_size + b_size - cnv_offset_hi;

    let scratch = scratch.borrow();
    let (mut sum_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 2, res_dft_size);

    let mut preps = Vec::with_capacity(pis.len());
    let mut rest = scratch_1;
    for (mask, pi) in masks.iter().zip(pis) {
        ckks_ensure!(
            mask.size() == a_size && pi.size() == b_size,
            "{OP}: inconsistent operand sizes"
        );
        let (mut b_prep, next) = rest.take_cnv_pvec_right_scratch(module, 1, b_size);
        rest = next
            .apply_mut(|s| module.cnv_prepare_right(&mut b_prep, GLWEToBackendRef::<BE>::to_backend_ref(pi).data(), b_mask, s));
        preps.push(b_prep);
    }
    // for (i, mask) in masks.iter().enumerate() {
    //     let candidate_base = (i / 4) * 4;
    //     let band = i % 4;
    //     let pi_idx = candidate_base + band_order[band];
    //     let pi = &pis[pi_idx];

    //     ckks_ensure!(
    //         mask.size() == a_size && pi.size() == b_size,
    //         "{OP}: inconsistent operand sizes"
    //     );

    //     let (mut b_prep, next) = rest.take_cnv_pvec_right_scratch(module, 1, b_size);

    //     rest = next
    //         .apply_mut(|s| module.cnv_prepare_right(&mut b_prep, GLWEToBackendRef::<BE>::to_backend_ref(pi).data(), b_mask, s));

    //     preps.push(b_prep);
    // }

    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        // for col in 0..2 {
        //     let terms: Vec<CnvDftAccTerm<'_, BE>> = masks
        //         .iter()
        //         .zip(&preps)
        //         .map(|(mask, prep)| CnvDftAccTerm {
        //             a: mask.to_backend_ref(),
        //             a_col: col,
        //             b: prep.to_backend_ref(),
        //             b_col: 0,
        //         })
        //         .collect();
        //     module.cnv_accumulate_dft(cnv_offset_hi, &mut sum_dft_mut, col, &terms, &mut rest.borrow());
        // }
        for col in 0..2 {
            let terms: Vec<CnvDftAccTerm<'_, BE>> = masks
                .iter()
                .enumerate()
                .map(|(i, mask)| {
                    let candidate_base = (i / 4) * 4;
                    let band = i % 4;
                    let prep_idx = candidate_base + band_order[band];
                    let prep = &preps[prep_idx];

                    CnvDftAccTerm {
                        a: mask.to_backend_ref(),
                        a_col: col,
                        b: prep.to_backend_ref(),
                        b_col: 0,
                    }
                })
                .collect();

            module.cnv_accumulate_dft(cnv_offset_hi, &mut sum_dft_mut, col, &terms, &mut rest.borrow());
        }
    }

    let (mut res_big, mut scratch_2) = rest.take_vec_znx_big_scratch(module, 2, res_dft_size);
    {
        let mut res_big_mut = res_big.to_backend_mut();
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_dft_mut, col);
        }
    }
    acc.set_log_budget(res_log_budget);
    acc.set_log_delta(res_log_delta);
    let res_big_ref = res_big.to_backend_ref();
    {
        let mut acc_mut = acc.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                acc_mut.data_mut(),
                base2k,
                res_log_budget + res_log_delta,
                cnv_offset_lo,
                col,
                &res_big_ref,
                base2k,
                col,
                &mut scratch_2.borrow(),
            );
        }
    }
    Ok(())
}

/// Complex SHIP masking with one shared RHS preparation.
///
/// The omega_1 / real path uses:
///     [pi1, pi2, pi3, pi4]
///
/// The omega_2 / imaginary path reuses the same encrypted masks with:
///     [pi3, pi4, pi2, pi1]
///
/// The plaintext RHS operands are prepared only once and shared by both paths.
pub(crate) fn ship_masking_accumulate_dual<BE>(
    module: &Module<BE>,
    acc_real: &mut CKKSCiphertextOwned<BE>,
    acc_imag: &mut CKKSCiphertextOwned<BE>,
    plan: &ShipPlan,
    masks: &[CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>],
    pis: &[CKKSPlaintextOwned<BE>],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_masking_accumulate_dual";

    ckks_ensure!(
        !masks.is_empty() && masks.len() == pis.len() && masks.len() % 4 == 0,
        "{OP}: empty, mismatched, or malformed operands"
    );

    // Both output ciphertexts must have the same physical representation.
    ckks_ensure!(
        acc_real.base2k().as_usize() == acc_imag.base2k().as_usize()
            && acc_real.max_k().as_usize() == acc_imag.max_k().as_usize(),
        "{OP}: real/imag output layouts do not match"
    );

    let base2k = acc_real.base2k().as_usize();
    let kk = plan.raised_k(base2k);
    let ld = plan.log_delta_work();

    let (res_log_budget, res_log_delta, cnv_offset) = mul_pt_params_raw(
        acc_real.max_k().as_usize(),
        ld,
        kk - ld,
        pis[0].log_delta(),
        pis[0].log_budget(),
        pis[0].max_k().as_usize(),
    )?;

    let a_size = kk.div_ceil(base2k);
    let b_size = pis[0].size();
    let b_mask = msb_mask_bottom_limb(base2k, pis[0].max_k().as_usize());

    let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < base2k {
        (0, -((base2k - (cnv_offset % base2k)) as i64))
    } else {
        ((cnv_offset / base2k).saturating_sub(1), (cnv_offset % base2k) as i64)
    };

    let res_dft_size = a_size + b_size - cnv_offset_hi;

    /*
     * Scratch layout:
     *
     *   sum_dft
     *   prepared(pi1 ... pi_{4 theta})
     *   reusable work area
     *
     * The work area is reused sequentially by:
     *   - real convolution
     *   - real IDFT/normalize
     *   - imag convolution
     *   - imag IDFT/normalize
     *
     * Therefore ship_masking_tmp_bytes() does not need to grow.
     */
    let scratch = scratch.borrow();

    let (mut sum_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 4, res_dft_size);

    let mut rest = scratch_1;

    // ------------------------------------------------------------
    // Phase A: prepare every pi exactly once, in canonical order.
    // ------------------------------------------------------------

    let mut preps = Vec::with_capacity(pis.len());

    for (mask, pi) in masks.iter().zip(pis) {
        ckks_ensure!(
            mask.size() == a_size && pi.size() == b_size,
            "{OP}: inconsistent operand sizes"
        );

        let (mut b_prep, next) = rest.take_cnv_pvec_right_scratch(module, 1, b_size);

        rest = next
            .apply_mut(|s| module.cnv_prepare_right(&mut b_prep, GLWEToBackendRef::<BE>::to_backend_ref(pi).data(), b_mask, s));

        preps.push(b_prep);
    }

    // real: [pi1, pi2, pi3, pi4]
    //
    // imag: [pi3, pi4, pi2, pi1]
    //
    // Indices are zero based.
    const BAND_ORDERS: [[usize; 4]; 2] = [[0, 1, 2, 3], [2, 3, 1, 0]];

    // ------------------------------------------------------------
    // Phase B: evaluate the real/imag convolution sums together.
    // ------------------------------------------------------------
    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            let terms_real: Vec<CnvDftAccTerm<'_, BE>> = masks
                .iter()
                .enumerate()
                .map(|(i, mask)| {
                    let candidate_base = (i / 4) * 4;
                    let band = i % 4;
                    let prep = &preps[candidate_base + BAND_ORDERS[0][band]];
                    CnvDftAccTerm {
                        a: mask.to_backend_ref(),
                        a_col: col,
                        b: prep.to_backend_ref(),
                        b_col: 0,
                    }
                })
                .collect();
            let terms_imag: Vec<CnvDftAccTerm<'_, BE>> = masks
                .iter()
                .enumerate()
                .map(|(i, mask)| {
                    let candidate_base = (i / 4) * 4;
                    let band = i % 4;
                    let prep = &preps[candidate_base + BAND_ORDERS[1][band]];
                    CnvDftAccTerm {
                        a: mask.to_backend_ref(),
                        a_col: col,
                        b: prep.to_backend_ref(),
                        b_col: 0,
                    }
                })
                .collect();
            module.cnv_accumulate_dft_dual(
                cnv_offset_hi,
                &mut sum_dft_mut,
                col,
                &terms_real,
                col + 2,
                &terms_imag,
                &mut rest.borrow(),
            );
        }
    }

    // ------------------------------------------------------------
    // Phase C: IDFT/normalize each half; res_big is reused.
    // ------------------------------------------------------------
    for half in 0..2 {
        let src_base = 2 * half;
        let (mut res_big, mut scratch_2) = rest.borrow().take_vec_znx_big_scratch(module, 2, res_dft_size);
        {
            let mut res_big_mut = res_big.to_backend_mut();
            let mut sum_dft_mut = sum_dft.to_backend_mut();
            for col in 0..2 {
                module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_dft_mut, src_base + col);
            }
        }
        let acc: &mut CKKSCiphertextOwned<BE> = if half == 0 { &mut *acc_real } else { &mut *acc_imag };
        acc.set_log_budget(res_log_budget);
        acc.set_log_delta(res_log_delta);
        let res_big_ref = res_big.to_backend_ref();
        let mut acc_mut = acc.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                acc_mut.data_mut(),
                base2k,
                res_log_budget + res_log_delta,
                cnv_offset_lo,
                col,
                &res_big_ref,
                base2k,
                col,
                &mut scratch_2.borrow(),
            );
        }
    }

    Ok(())
}
