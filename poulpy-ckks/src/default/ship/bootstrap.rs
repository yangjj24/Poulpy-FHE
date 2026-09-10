//! Backend-generic SHIP half-bootstrap circuit (Algorithm 1).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef;
use poulpy_core::{
    GLWEKeyswitch, GLWEZero,
    default::keyswitching::glwe::GGLWEProductDefault,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApplyTmpA,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    masking::{ship_masking_accumulate, ship_masking_accumulate_dual},
    mux::{ship_mux_plans, ship_mux_rotate, ship_mux_rotate_dual},
};
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSMulOps, CKKSSubOps, ShipScalar},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, ShipKeysPrepared},
    oep::{CKKSEncodingImpl, CKKSShipCoeffEncodingImpl},
};

/// Validates the runtime ciphertexts against the key bundle's parameters.
pub(crate) fn validate_runtime<BE, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    complex: bool,
) -> Result<()>
where
    BE: Backend,
    Src: CKKSCtBounds,
{
    const OP: &str = "ckks_ship_bootstrap";
    let params = keys.parameters();
    let plan = params.plan();
    let base2k = params.base2k();
    ckks_ensure!(
        module.n() == plan.n(),
        "{OP}: module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    ckks_ensure!(
        !complex || params.complex(),
        "{OP}: keys were not generated for complex bootstrap"
    );
    ckks_ensure!(
        input.n().as_usize() == plan.n() && input.rank().as_usize() == 1,
        "{OP}: input degree {} / rank {} does not match the plan",
        input.n(),
        input.rank()
    );
    ckks_ensure!(
        input.base2k().as_usize() == base2k && input.k().as_usize() == base2k,
        "{OP}: input must span a single limb of base2k {base2k}, got base2k {} and width {}",
        input.base2k(),
        input.k()
    );
    ckks_ensure!(
        output.n().as_usize() == plan.n() && output.rank().as_usize() == 1,
        "{OP}: output degree {} / rank {} does not match the plan",
        output.n(),
        output.rank()
    );
    ckks_ensure!(
        output.base2k().as_usize() == base2k,
        "{OP}: output base2k {} does not match the key radix {base2k}",
        output.base2k()
    );
    ckks_ensure!(
        output.max_k().as_usize() >= plan.raised_k(base2k),
        "{OP}: output capacity {} is below the raised precision {}",
        output.max_k(),
        plan.raised_k(base2k)
    );
    Ok(())
}

/// Shared core of both bootstrap entry points: encapsulation, masking, blind
/// rotations and product tree per coefficient half, returning the product-tree
/// roots in half order.
fn ship_bootstrap_roots<BE, F, Src>(
    module: &Module<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    complex: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<CKKSCiphertextOwned<BE>>>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + GGLWEProductDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    const OP: &str = "ckks_ship_bootstrap";
    let params = keys.parameters();
    let plan = *params.plan();
    let base2k = params.base2k();
    let b2k_t: poulpy_core::layouts::Base2K = base2k.into();
    let kk = plan.raised_k(base2k);
    let ld = plan.log_delta_work();
    let theta = plan.theta();
    let halves = if complex { 2 } else { 1 };

    // Encapsulation: switch the bottom ciphertext from the dense to the
    // sparse secret at the bottom modulus.
    let mut a_sparse = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(base2k as u32));
    module.glwe_keyswitch(&mut a_sparse, input, &keys.dense_to_sparse().to_backend_ref(), scratch);
    a_sparse.set_meta_checked(input.meta())?;

    let enc = BE::ckks_ship_coeff_encodings_impl::<F, _>(module, &a_sparse, &plan, b2k_t, complex, scratch)?;
    ckks_ensure!(
        enc.pi.len() == plan.sparse_hamming_weight() && (!complex || enc.pt0_2.is_some()),
        "{OP}: malformed coefficient encodings"
    );

    // Leaf 0 per half: trivial encryption of pt0 / pt0_2.
    let mut leaves: Vec<Vec<CKKSCiphertextOwned<BE>>> = Vec::with_capacity(halves);
    for half in 0..halves {
        let pt0 = if half == 0 {
            &enc.pt0
        } else {
            enc.pt0_2.as_ref().expect("complex encodings carry pt0_2")
        };
        let mut leaf0 = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));
        module.glwe_zero(&mut leaf0);
        leaf0.set_meta_checked(CKKSMeta {
            log_delta: ld,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        })?;
        module.ckks_add_pt_vec_assign(&mut leaf0, pt0, scratch)?;
        let mut half_leaves = Vec::with_capacity(plan.sparse_hamming_weight() + 1);
        half_leaves.push(leaf0);
        leaves.push(half_leaves);
    }

    // Leaves 1..=h: theta-column masking followed by the hoisted
    // base-B mux blind rotation.
    //
    // In the complex path, both coefficient halves share the same
    // encrypted selector masks and the same canonical preparation of
    // the pi plaintexts.  The omega_2 path differs only by the fixed
    // per-quartet pi permutation [2, 3, 1, 0].
    //
    // H-MUX keys are shared and traversed in lockstep by the dual path.
    let plans = ship_mux_plans(
        module,
        keys.index_keys()
            .iter()
            .flat_map(|ik| ik.mux_keys().iter().map(Vec::as_slice)),
    );

    for (slot, ik) in keys.index_keys().iter().enumerate() {
        let pi = &enc.pi[slot];

        ckks_ensure!(pi.len() == 4 * theta, "{OP}: malformed pi encodings at slot {slot}");

        if complex {
            // Both coefficient halves reuse the same encrypted masks and
            // the same canonical preparation of pi.
            let mut acc_real = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));

            let mut acc_imag = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));

            ship_masking_accumulate_dual(module, &mut acc_real, &mut acc_imag, &plan, ik.masks(), pi, scratch)?;

            // The two complex coefficient halves use the same H-MUX key group.
            // Traverse each prepared key once and evaluate both products together.
            for group in ik.mux_keys() {
                ship_mux_rotate_dual(module, &mut acc_real, &mut acc_imag, group, &plans, scratch)?;
            }

            leaves[0].push(acc_real);
            leaves[1].push(acc_imag);
        } else {
            let mut acc = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));

            ship_masking_accumulate(module, &mut acc, &plan, ik.masks(), pi, [0, 1, 2, 3], scratch)?;

            for group in ik.mux_keys() {
                ship_mux_rotate(module, &mut acc, group, &plans, scratch)?;
            }

            leaves[0].push(acc);
        }
    }

    // Binary product tree. In the complex case the two trees have identical
    // topology, so matching multiplications are evaluated in lockstep and
    // share tensor-key relinearization. The tensor products themselves remain
    // independent, avoiding real/imag cross terms.
    let roots = if complex {
        ckks_ensure!(
            leaves.len() == 2 && leaves[0].len() == leaves[1].len(),
            "{OP}: malformed complex product leaves"
        );
        let mut imag = leaves.pop().expect("imag leaves exist");
        let mut real = leaves.pop().expect("real leaves exist");

        while real.len() > 1 {
            ckks_ensure!(real.len() == imag.len(), "{OP}: complex product trees diverged");
            let mut next_real = Vec::with_capacity(real.len().div_ceil(2));
            let mut next_imag = Vec::with_capacity(imag.len().div_ceil(2));
            let mut real_iter = real.into_iter();
            let mut imag_iter = imag.into_iter();

            while let (Some(xr), Some(xi)) = (real_iter.next(), imag_iter.next()) {
                match (real_iter.next(), imag_iter.next()) {
                    (Some(yr), Some(yi)) => {
                        let budget_r = xr.log_budget().min(yr.log_budget());
                        let consumed_r = xr.log_delta().max(yr.log_delta());
                        let budget_i = xi.log_budget().min(yi.log_budget());
                        let consumed_i = xi.log_delta().max(yi.log_delta());
                        ckks_ensure!(
                            budget_r >= consumed_r && budget_i >= consumed_i,
                            "{OP}: product tree exhausts the budget"
                        );
                        let k_dst_r = budget_r - consumed_r + xr.log_delta().min(yr.log_delta());
                        let k_dst_i = budget_i - consumed_i + xi.log_delta().min(yi.log_delta());
                        ckks_ensure!(k_dst_r == k_dst_i, "{OP}: complex product tree precisions diverged");

                        let mut dst_r = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(k_dst_r as u32));
                        let mut dst_i = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(k_dst_i as u32));
                        module.ckks_mul_into_dual(&mut dst_r, &xr, &yr, &mut dst_i, &xi, &yi, keys.tensor_key(), scratch)?;
                        next_real.push(dst_r);
                        next_imag.push(dst_i);
                    }
                    (None, None) => {
                        next_real.push(xr);
                        next_imag.push(xi);
                    }
                    _ => return Err(anyhow::anyhow!("{OP}: complex product tree pairing diverged").into()),
                }
            }
            ckks_ensure!(
                real_iter.next().is_none() && imag_iter.next().is_none(),
                "{OP}: complex product trees diverged"
            );
            real = next_real;
            imag = next_imag;
        }
        vec![
            real.pop().expect("real product tree is never empty"),
            imag.pop().expect("imag product tree is never empty"),
        ]
    } else {
        let mut level = leaves.pop().expect("real product leaves exist");
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut iter = level.into_iter();
            while let Some(x) = iter.next() {
                match iter.next() {
                    Some(y) => {
                        let budget = x.log_budget().min(y.log_budget());
                        let consumed = x.log_delta().max(y.log_delta());
                        ckks_ensure!(budget >= consumed, "{OP}: product tree exhausts the budget");
                        let k_dst = budget - consumed + x.log_delta().min(y.log_delta());
                        let mut dst = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(k_dst as u32));
                        module.ckks_mul_into(&mut dst, &x, &y, keys.tensor_key(), scratch)?;
                        next.push(dst);
                    }
                    None => next.push(x),
                }
            }
            level = next;
        }
        vec![level.pop().expect("product tree is never empty")]
    };
    Ok(roots)
}

/// Real-case SHIP bootstrap: `output = root + Conj(root)`.
pub(crate) fn ship_bootstrap_into<BE, F, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + GGLWEProductDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    validate_runtime(module, output, input, keys, false)?;
    let base2k: poulpy_core::layouts::Base2K = keys.parameters().base2k().into();
    let mut roots = ship_bootstrap_roots::<BE, F, _>(module, input, keys, false, scratch)?;

    let root = roots.pop().expect("real bootstrap has one root");
    let mut conj = module.ckks_ciphertext_alloc(base2k, root.k());
    module.ckks_conjugate_into(&mut conj, &root, keys.conjugation_key(), scratch)?;
    module.ckks_add_into(output, &root, &conj, scratch)?;
    // `root + conj(root) = 2·Re(root)`.
    output.set_slots(SlotsKind::Real);
    Ok(())
}

/// Complex-case SHIP bootstrap:
/// `output = (v1 + i*v2) + Conj(v1 - i*v2)` over the two per-half roots.
pub(crate) fn ship_bootstrap_complex_into<BE, F, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + GGLWEProductDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    validate_runtime(module, output, input, keys, true)?;
    let base2k: poulpy_core::layouts::Base2K = keys.parameters().base2k().into();
    let mut roots = ship_bootstrap_roots::<BE, F, _>(module, input, keys, true, scratch)?;

    let v2 = roots.pop().expect("complex bootstrap has two roots");
    let v1 = roots.pop().expect("complex bootstrap has two roots");
    let k_eff = v1.k();
    let mut iv2 = module.ckks_ciphertext_alloc(base2k, v2.k());
    module.ckks_mul_i_into(&mut iv2, &v2, scratch)?;
    let mut w_plus = module.ckks_ciphertext_alloc(base2k, k_eff);
    module.ckks_add_into(&mut w_plus, &v1, &iv2, scratch)?;
    let mut w_minus = module.ckks_ciphertext_alloc(base2k, k_eff);
    module.ckks_sub_into(&mut w_minus, &v1, &iv2, scratch)?;
    let mut conj = module.ckks_ciphertext_alloc(base2k, w_minus.k());
    module.ckks_conjugate_into(&mut conj, &w_minus, keys.conjugation_key(), scratch)?;
    module.ckks_add_into(output, &w_plus, &conj, scratch)?;
    Ok(())
}
