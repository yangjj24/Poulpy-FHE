//! SHIP bootstrapping tests (host replica, HMuxRot primitive, end-to-end).

use poulpy_core::layouts::GLWEInfos as _;
use std::f64::consts::TAU;

use poulpy_core::{
    GLWENormalizeDefault, GLWEZero, TransferInto,
    default::keyswitching::glwe::GGLWEProductDefault,
    layouts::{
        BackendGLWESecret, GLWESecretPreparedFactory, GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, CnvPVecBytesOf, Convolution, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftApply,
        VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApplyTmpA, VecZnxNormalize,
        VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{CnvPVecLToBackendMut, HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSInfos, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEncodingOps, CKKSEncryptOps, CKKSShipOps, ShipScalar},
    encoding::ship::masks::{ship_pre_rotated_masks, ship_pre_rotated_masks_omega2},
    layouts::CKKSPlaintextVecHostCodec,
    layouts::{
        CKKSEncodingBuffer, CKKSModuleAlloc, ShipKeySet, ShipKeysLayout, ShipPlan, ShipSecretSpec,
        ship::keyset::{HMuxRotKeyPrepared, hmux_rot_key_encrypt_sk, hmux_rot_packed_key_encrypt_sk, ship_sheared_packed_beta},
    },
    oep::{CKKSEncodingImpl, CKKSShipCoeffEncodingImpl},
    test_suite::CKKSTestParams,
};

use super::helpers::{
    TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
    ckks_decrypt_decode, ckks_encrypt, ckks_encrypt_pt, ckks_spec, test_vector_1, want_rotate,
};
use poulpy_core::layouts::GLWESecretSampling;

/// SHIP suite plan on the test ring: sparse weight 32, full offset window,
/// base-4 mux with the low digit absorbed into masking.
fn ship_suite_plan(params: &CKKSTestParams) -> ShipPlan {
    const SPARSE_HW: usize = 32;
    let window = (params.n / SPARSE_HW) / 2;
    ShipPlan::new(
        params.log_n(),
        6,
        // ~14 bits of CKKS noise accumulate below log_delta_work; 30+ keeps
        // the gap model (~12.7 bits for gamma = 2^6) dominant.
        params.prec_meta.log_delta.min(40),
        2 * params.base2k,
        SPARSE_HW,
        window,
        4,
        4,
    )
    .unwrap()
}

/// Target sheared-SHIP regression plan: sparse weight h=31 gives exactly
/// 31 secret factors plus one public pt0 factor, hence G=32 with no padding.
/// This stays separate from ship_suite_plan so established h=32 baseline
/// tests remain unchanged.
fn ship_sheared_suite_plan(params: &CKKSTestParams) -> ShipPlan {
    const SPARSE_HW: usize = 31;
    let window = (params.n / SPARSE_HW) / 2;
    ShipPlan::new(
        params.log_n(),
        6,
        params.prec_meta.log_delta.min(40),
        2 * params.base2k,
        SPARSE_HW,
        window,
        4,
        4,
    )
    .unwrap()
}

/// Like `gen_sk_with_raw` but also returns the host-side secret.
#[allow(clippy::type_complexity)]
fn gen_sk_with_host<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    seed: [u8; 32],
) -> (
    BackendGLWESecret<HostBytesBackend>,
    BackendGLWESecret<BE>,
    GLWESecretPrepared<BE::OwnedBuf, BE>,
)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    let glwe_infos = params.glwe_layout();
    let mut source = Source::new(seed);
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_infos);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, params.hw, &mut source);
    let mut sk_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_raw.transfer_into(&mut sk_host);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    (sk_host, sk_raw, sk)
}

/// Paper-convention slot rotation: `out[i] = in[(i - r) mod m]`.
fn rot_paper(v: &[(f64, f64)], r: usize) -> Vec<(f64, f64)> {
    let m = v.len();
    (0..m).map(|i| v[(i + m - (r % m)) % m]).collect()
}

fn cmul((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// Cleartext replica of SHIP Algorithm 1 extended to the full/complex case
/// (Lemma 4 omega_2 masks): settles the phase, conjugation, mask and
/// recombination conventions before any homomorphic code runs.
pub fn test_ship_host_replica<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
{
    let plan = ShipPlan::new(params.log_n(), 6, 40, 0, 32, (params.n / 32) / 2, 4, 4).unwrap();
    let n = plan.n();
    let m = plan.half_n();
    let theta = plan.theta();
    let w = plan.window();
    let base2k = params.base2k;
    let q0f = 2f64.powi(base2k as i32);
    let gamma = 2f64.powi(plan.log_gamma() as i32);
    let bases = plan.mux_bases();

    let mut source = Source::new([42u8; 32]);
    let spec = ShipSecretSpec::sample(&plan, &mut source);
    let mut s = vec![0i64; n];
    for &(idx, sign) in spec.support() {
        s[idx] = sign;
    }

    // Complex cleartext payload mu with gap gamma: Re in the first m
    // coefficients, Im in the last m.
    let mu: Vec<(f64, f64)> = (0..m)
        .map(|_| (source.next_f64(-0.45, 0.45), source.next_f64(-0.45, 0.45)))
        .collect();
    let mut m_int = vec![0i64; n];
    for i in 0..m {
        m_int[i] = (q0f * mu[i].0 / gamma).round() as i64;
        m_int[i + m] = (q0f * mu[i].1 / gamma).round() as i64;
    }
    let mu_quant: Vec<(f64, f64)> = (0..m)
        .map(|i| (gamma * m_int[i] as f64 / q0f, gamma * m_int[i + m] as f64 / q0f))
        .collect();

    // Bottom ciphertext: b = m - a*s mod (X^n + 1, q0).
    let q0i = 1i64 << base2k;
    let a: Vec<i64> = (0..n)
        .map(|_| source.next_u64n(q0i as u64, (q0i - 1) as u64) as i64)
        .collect();
    let mut b = m_int.clone();
    for (j, &sj) in s.iter().enumerate() {
        if sj == 0 {
            continue;
        }
        for (i, &ai) in a.iter().enumerate() {
            let e = i + j;
            let (idx, sign) = if e >= n { (e - n, -sj) } else { (e, sj) };
            b[idx] = (b[idx] - sign * ai).rem_euclid(q0i);
        }
    }

    // Exponential cleartexts. pt0 of each half carries gamma/(4*i*pi) and is
    // seeded with the matching half of b.
    let omega = |x: i64| -> (f64, f64) {
        let phi = TAU * (x.rem_euclid(q0i) as f64) / q0f;
        (phi.cos(), phi.sin())
    };
    let pt0 = |off: usize| -> Vec<(f64, f64)> {
        (0..m)
            .map(|i| {
                let (re, im) = omega(b[i + off]);
                (gamma / (2.0 * TAU) * im, -(gamma / (2.0 * TAU)) * re)
            })
            .collect()
    };
    let pi: [Vec<(f64, f64)>; 4] = [
        (0..m).map(|i| omega(a[i])).collect(),
        (0..m).map(|i| omega(-a[i])).collect(),
        (0..m).map(|i| omega(a[i + m])).collect(),
        (0..m).map(|i| omega(-a[i + m])).collect(),
    ];

    // Per support index and coefficient half: theta-column masking then the
    // mux digit rotations, with a per-index correctness check.
    let mut v = [pt0(0), pt0(m)];
    for (k, &(j, sj)) in spec.support().iter().enumerate() {
        let p = plan.mask_rotation(k);
        let u = spec.offset(&plan, k);
        assert!(u <= 2 * w, "ship_host_replica: support index outside window");

        let mask_sets = [ship_pre_rotated_masks(j, sj, m), ship_pre_rotated_masks_omega2(j, sj, m)];
        for (half, masks) in mask_sets.iter().enumerate() {
            // Column/mux hybrid (§4.4, low-digit-column variant): the low
            // digit u0 = u mod theta is absorbed into the masking via
            // selector masks 1_{c = u0} x Rot_{p+c}(mask), paired with pi
            // rotated by p + c.
            let u0 = u % theta;
            let ct_j: Vec<(f64, f64)> = (0..m)
                .map(|i| {
                    let mut acc = (0.0, 0.0);
                    for c in 0..theta {
                        if c != u0 {
                            continue;
                        }
                        let src = (i + 2 * m - (p + c)) % m;
                        for k4 in 0..4 {
                            acc.0 += pi[k4][src].0 * masks[k4][src];
                            acc.1 += pi[k4][src].1 * masks[k4][src];
                        }
                    }
                    acc
                })
                .collect();

            // Mux blind rotation over the mixed-radix digits of u / theta.
            let mut rotated = ct_j;
            let mut weight = theta;
            for &base in &bases {
                let digit = (u / weight) % base;
                rotated = rot_paper(&rotated, digit * weight);
                weight *= base;
            }
            assert!(weight > 2 * w, "ship_host_replica: mux chain does not cover the window");
            assert_eq!((p + u) % m, j % m, "ship_host_replica: rotation split mismatch");

            // rotated must equal (omega^{(s_j * a * X^j)_{i + half*m}})_{i < m}.
            for (i, got) in rotated.iter().enumerate() {
                let e = i + half * m + n - j;
                let (idx, sign) = if e >= n { (e - n, sj) } else { (e, -sj) };
                let want = omega(sign * a[idx % n]);
                let err = ((got.0 - want.0).abs()).max((got.1 - want.1).abs());
                assert!(
                    err < 1e-9,
                    "ship_host_replica: blind rotation half={half} j={j} s_j={sj} slot {i}: err={err:.3e}"
                );
            }

            v[half] = (0..m).map(|i| cmul(v[half][i], rotated[i])).collect();
        }
    }

    // z_h = v_h + conj(v_h) extracts each coefficient half into real slots;
    // out = z_1 + i*z_2 ~ mu. With one conjugation:
    // out = (v_1 + i*v_2) + conj(v_1 - i*v_2).
    let mut max_err: f64 = 0.0;
    for i in 0..m {
        let (w_re, w_im) = (v[0][i].0 - v[1][i].1, v[0][i].1 + v[1][i].0); // v1 + i*v2
        let (c_re, c_im) = (v[0][i].0 + v[1][i].1, -(v[0][i].1 - v[1][i].0)); // conj(v1 - i*v2)
        let out = (w_re + c_re, w_im + c_im);
        assert!(
            (out.0 - 2.0 * v[0][i].0).abs() < 1e-12 && (out.1 - 2.0 * v[1][i].0).abs() < 1e-12,
            "ship_host_replica: one-conjugation recombination mismatch"
        );
        max_err = max_err.max((out.0 - mu_quant[i].0).abs()).max((out.1 - mu_quant[i].1).abs());
    }
    // Model: (2*pi)^2 * mu^3 / (6 * gamma^2) ~ 1.5e-4 for |mu| <= 0.45.
    assert!(
        max_err < 3e-4,
        "ship_host_replica: final error {max_err:.3e} exceeds model bound"
    );
}

use poulpy_core::layouts::GLWESecretTensorFactory as _;
use poulpy_core::layouts::prepared::GLWESecretTensorPreparedFactory as _;
/// Hoisted B-to-1 mux-rotate: a digit-position key group with `beta` on digit
/// `d` yields the paper-convention rotation `out[i] = in[i - d*weight]`; an
/// all-zero group yields (an encryption of) zero.
use poulpy_core::{GLWEDecrypt as _, GLWETensorDecrypt as _, GLWETensoring as _};

pub fn test_ship_mux_rotate<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSEncryptOps<BE>
        + CKKSShipOps<BE, F>
        + CKKSAddOps<BE>
        + crate::api::CKKSConjugateOps<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf
        + GGLWEProductDefault<BE>
        + poulpy_core::GLWEZeroDefault<BE>
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
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxNormalizeTmpBytes
        + crate::api::CKKSRotateOps<BE>
        + poulpy_core::GLWETensoring<BE>
        + poulpy_core::GLWETensorDecrypt<BE>
        + poulpy_core::GLWEDecrypt<BE>
        + poulpy_core::layouts::GLWESecretTensorFactory<BE>
        + poulpy_core::layouts::prepared::GLWESecretTensorPreparedFactory<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
{
    let m = params.n / 2;
    let k = params.k;
    let encoder = super::reference_encoder::ReferenceEncoder::<E>::new::<F>(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_host, _, sk) = gen_sk_with_host(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut xe = Source::new([21u8; 32]);
    let mut xa = Source::new([22u8; 32]);
    // Exercise the multi-limb gadget path. On exact backends this also covers
    // the shortened accumulator used when several mux products share one
    // normalization.
    let mux_dsize = params.dsize.max(4);
    // Single-candidate correctness check for the experimental packed H-MUX.
    //
    // The selector is 32-periodic and the physical rotation is 32, so the
    // output automorphism preserves the selector lanes.
    const PACKED_BETA_SCALE_BITS: usize = 30;
    const GROUPS: usize = 32;
    assert!(m.is_multiple_of(GROUPS));

    let support_offsets: Vec<usize> = (0..(GROUPS - 1)).collect();
    let beta_slots = ship_sheared_packed_beta(&support_offsets, 1, 2, 0, m);
    let beta_re: Vec<F> = beta_slots
        .iter()
        .map(|&bit| F::from_f64(if bit { 1.0 } else { 0.0 }).unwrap())
        .collect();
    let beta_im = vec![F::zero(); m];
    let mut beta_coeffs = vec![F::zero(); params.n];
    encoder.pack_reim_coeffs(&mut beta_coeffs, &beta_re, &beta_im).unwrap();

    let beta_scale = (1u64 << PACKED_BETA_SCALE_BITS) as f64;
    let beta_hat: Vec<i64> = beta_coeffs
        .iter()
        .map(|x| (x.to_f64().unwrap() * beta_scale).round() as i64)
        .collect();

    let physical_rot = GROUPS;
    let rot = physical_rot % m;

    // Diagnostic A: exact all-one selector. In coefficient form the constant
    // polynomial 2^lambda evaluates to 2^lambda in every CKKS slot, so after
    // the +lambda normalization offset this must reduce to a plain Rot_32.
    let mut exact_beta_hat = vec![0i64; params.n];
    exact_beta_hat[0] = 1i64 << PACKED_BETA_SCALE_BITS;
    let mut exact_xe = Source::new([29u8; 32]);
    let mut exact_xa = Source::new([30u8; 32]);
    let exact_key = hmux_rot_packed_key_encrypt_sk(
        module,
        host_module,
        &sk_host,
        &exact_beta_hat,
        physical_rot,
        k,
        params.base2k.into(),
        mux_dsize,
        &mut exact_xe,
        &mut exact_xa,
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut exact_prepared = module.glwe_switching_key_prepared_alloc_from_infos(exact_key.key());
    module.glwe_switching_key_prepare(&mut exact_prepared, exact_key.key(), &mut scratch.borrow());
    let exact_group = vec![HMuxRotKeyPrepared {
        key: exact_prepared,
        gal_el: exact_key.gal_el(),
    }];

    let mut exact_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let exact_plans = crate::default::ship::mux::ship_mux_plans(module, std::iter::once(exact_group.as_slice()));
    let exact_mux_bytes =
        crate::default::ship::mux::ship_mux_rotate_tmp_bytes(module, &exact_ct, &exact_group[0].key, exact_group.len());
    let mut exact_mux_scratch = ScratchOwned::<BE>::alloc(exact_mux_bytes);
    crate::default::ship::mux::ship_mux_rotate_with_offset(
        module,
        &mut exact_ct,
        &exact_group,
        &exact_plans,
        -(PACKED_BETA_SCALE_BITS as i64),
        &mut exact_mux_scratch.borrow(),
    )
    .unwrap();

    let (exact_re, exact_im) = ckks_decrypt_decode(&params, module, &encoder, &exact_ct, &sk, &mut scratch.borrow());
    let mut exact_max_err = 0.0f64;
    let mut exact_sq_err = 0.0f64;
    for i in 0..m {
        let er = (exact_re[i].to_f64().unwrap() - re1[(i + m - rot) % m].to_f64().unwrap()).abs();
        let ei = (exact_im[i].to_f64().unwrap() - im1[(i + m - rot) % m].to_f64().unwrap()).abs();
        exact_max_err = exact_max_err.max(er).max(ei);
        exact_sq_err += er * er + ei * ei;
    }
    let exact_rms = (exact_sq_err / (2 * m) as f64).sqrt();
    let exact_max_bits = if exact_max_err == 0.0 {
        f64::INFINITY
    } else {
        -exact_max_err.log2()
    };
    let exact_rms_bits = if exact_rms == 0.0 { f64::INFINITY } else { -exact_rms.log2() };
    println!(
        "[packed-hmux diag] backend={} exact-one: max_err={:.3e} ({:.1} bits), rms={:.3e} ({:.1} bits)",
        std::any::type_name::<BE>(),
        exact_max_err,
        exact_max_bits,
        exact_rms,
        exact_rms_bits,
    );

    let mut packed_xe = Source::new([31u8; 32]);
    let mut packed_xa = Source::new([32u8; 32]);
    let packed_key = hmux_rot_packed_key_encrypt_sk(
        module,
        host_module,
        &sk_host,
        &beta_hat,
        physical_rot,
        k,
        params.base2k.into(),
        mux_dsize,
        &mut packed_xe,
        &mut packed_xa,
        &mut scratch.borrow(),
    )
    .unwrap();
    let mut packed_prepared = module.glwe_switching_key_prepared_alloc_from_infos(packed_key.key());
    module.glwe_switching_key_prepare(&mut packed_prepared, packed_key.key(), &mut scratch.borrow());
    let packed_group = vec![HMuxRotKeyPrepared {
        key: packed_prepared,
        gal_el: packed_key.gal_el(),
    }];

    let mut packed_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let packed_plans = crate::default::ship::mux::ship_mux_plans(module, std::iter::once(packed_group.as_slice()));
    let packed_mux_bytes =
        crate::default::ship::mux::ship_mux_rotate_tmp_bytes(module, &packed_ct, &packed_group[0].key, packed_group.len());
    let mut packed_mux_scratch = ScratchOwned::<BE>::alloc(packed_mux_bytes);
    crate::default::ship::mux::ship_mux_rotate_with_offset(
        module,
        &mut packed_ct,
        &packed_group,
        &packed_plans,
        -(PACKED_BETA_SCALE_BITS as i64),
        &mut packed_mux_scratch.borrow(),
    )
    .unwrap();

    let want_re: Vec<F> = (0..m)
        .map(|i| if beta_slots[i] { re1[(i + m - rot) % m] } else { F::zero() })
        .collect();
    let want_im: Vec<F> = (0..m)
        .map(|i| if beta_slots[i] { im1[(i + m - rot) % m] } else { F::zero() })
        .collect();

    // Diagnostic B: decode the non-trivial packed selector result directly.
    // This deliberately bypasses assert_decrypt_precision's full-width ring
    // budget check, because beta_hat itself is only a lambda-bit fixed-point
    // approximation.
    let (packed_re, packed_im) = ckks_decrypt_decode(&params, module, &encoder, &packed_ct, &sk, &mut scratch.borrow());
    let mut packed_max_err = 0.0f64;
    let mut packed_sq_err = 0.0f64;
    for i in 0..m {
        let er = (packed_re[i].to_f64().unwrap() - want_re[i].to_f64().unwrap()).abs();
        let ei = (packed_im[i].to_f64().unwrap() - want_im[i].to_f64().unwrap()).abs();
        packed_max_err = packed_max_err.max(er).max(ei);
        packed_sq_err += er * er + ei * ei;
    }
    let packed_rms = (packed_sq_err / (2 * m) as f64).sqrt();
    let packed_max_bits = if packed_max_err == 0.0 {
        f64::INFINITY
    } else {
        -packed_max_err.log2()
    };
    let packed_rms_bits = if packed_rms == 0.0 {
        f64::INFINITY
    } else {
        -packed_rms.log2()
    };
    println!(
        "[packed-hmux diag] backend={} packed-beta: max_err={:.3e} ({:.1} bits), rms={:.3e} ({:.1} bits)",
        std::any::type_name::<BE>(),
        packed_max_err,
        packed_max_bits,
        packed_rms,
        packed_rms_bits,
    );

    // ---------------------------------------------------------------------
    // Phase 3B: one complete packed digit -> one sheared output group.
    //
    // This is deliberately correctness-first: each candidate is evaluated
    // with the already validated single-source packed H-MUX, then the four
    // ciphertext terms are accumulated. A later phase will fuse the four
    // DFT/VMP paths before the final IDFT/normalize.
    // ---------------------------------------------------------------------
    {
        use crate::api::CKKSAddOps;

        const PHASE3B_BASE: usize = 4;
        const PHASE3B_WEIGHT: usize = 5;
        const PHASE3B_OUT_GROUP: usize = 7;
        const PHASE3B_MIN_PREC_BITS: f64 = 18.0;

        // Synthetic branch table x_r[j]. The values depend on both the
        // branch/lane r and logical index j so a wrong source-group route
        // cannot accidentally look correct.
        let branch_re: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((17 * r + 13 * j + 7) % 101) as f64;
                        F::from_f64((raw - 50.0) / 64.0).unwrap()
                    })
                    .collect()
            })
            .collect();
        let branch_im: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((11 * r + 19 * j + 3) % 97) as f64;
                        F::from_f64((raw - 48.0) / 96.0).unwrap()
                    })
                    .collect()
            })
            .collect();

        let phase3b_offsets: Vec<usize> = (0..(GROUPS - 1)).collect();
        let mut want_re = vec![F::zero(); m];
        let mut want_im = vec![F::zero(); m];
        let mut selector_hits = vec![0usize; m];
        let mut routes = Vec::with_capacity(PHASE3B_BASE);
        let mut acc: Option<crate::layouts::CKKSCiphertextOwned<BE>> = None;

        for candidate in 0..PHASE3B_BASE {
            let runtime_rot = candidate * PHASE3B_WEIGHT;
            let route = crate::default::ship::mux::ship_sheared_runtime_route_from_output(PHASE3B_OUT_GROUP, runtime_rot, GROUPS);
            routes.push((route.group, route.physical_rot));

            // Build the source sheared group:
            //     Y_s[p] = x_{p mod G}[(p + s) mod m].
            let src_re: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_re[lane][(p + route.group) % m]
                })
                .collect();
            let src_im: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_im[lane][(p + route.group) % m]
                })
                .collect();

            // Packed candidate selector Beta_d, including the public lane 31
            // convention (candidate zero).
            let beta_slots = ship_sheared_packed_beta(&phase3b_offsets, PHASE3B_WEIGHT, PHASE3B_BASE, candidate, m);
            let beta_re: Vec<F> = beta_slots
                .iter()
                .map(|&bit| F::from_f64(if bit { 1.0 } else { 0.0 }).unwrap())
                .collect();
            let beta_im = vec![F::zero(); m];
            let mut beta_coeffs = vec![F::zero(); params.n];
            encoder.pack_reim_coeffs(&mut beta_coeffs, &beta_re, &beta_im).unwrap();
            let beta_hat: Vec<i64> = beta_coeffs
                .iter()
                .map(|x| (x.to_f64().unwrap() * beta_scale).round() as i64)
                .collect();

            // Clear reference for this candidate. Under Poulpy's runtime
            // convention Rot_t(x)[p] = x[p-t], the sheared route gives:
            //
            //   Rot_rho(Y_s)[p] = x_lane[p + g - t].
            for p in 0..m {
                if beta_slots[p] {
                    selector_hits[p] += 1;
                    let lane = p % GROUPS;
                    let logical = (p + PHASE3B_OUT_GROUP + m - (runtime_rot % m)) % m;
                    want_re[p] = branch_re[lane][logical];
                    want_im[p] = branch_im[lane][logical];
                }
            }

            let mut term_xe = Source::new([40u8 + candidate as u8; 32]);
            let mut term_xa = Source::new([50u8 + candidate as u8; 32]);
            let term_key = hmux_rot_packed_key_encrypt_sk(
                module,
                host_module,
                &sk_host,
                &beta_hat,
                route.physical_rot,
                k,
                params.base2k.into(),
                mux_dsize,
                &mut term_xe,
                &mut term_xa,
                &mut scratch.borrow(),
            )
            .unwrap();

            let mut term_prepared = module.glwe_switching_key_prepared_alloc_from_infos(term_key.key());
            module.glwe_switching_key_prepare(&mut term_prepared, term_key.key(), &mut scratch.borrow());
            let term_group = vec![HMuxRotKeyPrepared {
                key: term_prepared,
                gal_el: term_key.gal_el(),
            }];

            let mut term_ct = ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                k,
                &src_re,
                &src_im,
                &mut scratch.borrow(),
            );
            let term_plans = crate::default::ship::mux::ship_mux_plans(module, std::iter::once(term_group.as_slice()));
            let term_mux_bytes =
                crate::default::ship::mux::ship_mux_rotate_tmp_bytes(module, &term_ct, &term_group[0].key, term_group.len());
            let mut term_mux_scratch = ScratchOwned::<BE>::alloc(term_mux_bytes);
            crate::default::ship::mux::ship_mux_rotate_with_offset(
                module,
                &mut term_ct,
                &term_group,
                &term_plans,
                -(PACKED_BETA_SCALE_BITS as i64),
                &mut term_mux_scratch.borrow(),
            )
            .unwrap();

            if let Some(acc_ct) = acc.as_mut() {
                module.ckks_add_assign(acc_ct, &term_ct, &mut scratch.borrow()).unwrap();
            } else {
                acc = Some(term_ct);
            }
        }

        // For all 31 sparse-secret lanes plus the public lane, exactly one
        // base-4 candidate must be active.
        assert!(
            selector_hits.iter().all(|&hits| hits == 1),
            "Phase 3B packed selectors do not form a one-hot partition"
        );

        // This concrete choice intentionally exercises both rho=0 and rho=32.
        assert_eq!(
            routes,
            vec![(7usize, 0usize), (2, 0), (29, 32), (24, 32)],
            "Phase 3B did not exercise the expected source/physical routes"
        );

        let acc = acc.expect("Phase 3B has four candidates");
        let (got_re, got_im) = ckks_decrypt_decode(&params, module, &encoder, &acc, &sk, &mut scratch.borrow());

        let mut max_err = 0.0f64;
        let mut sq_err = 0.0f64;
        for p in 0..m {
            let er = (got_re[p].to_f64().unwrap() - want_re[p].to_f64().unwrap()).abs();
            let ei = (got_im[p].to_f64().unwrap() - want_im[p].to_f64().unwrap()).abs();
            max_err = max_err.max(er).max(ei);
            sq_err += er * er + ei * ei;
        }
        let rms = (sq_err / (2 * m) as f64).sqrt();
        let max_bits = if max_err == 0.0 { f64::INFINITY } else { -max_err.log2() };
        let rms_bits = if rms == 0.0 { f64::INFINITY } else { -rms.log2() };

        println!(
            "[sheared phase3b] backend={} routes={:?} max_err={:.3e} ({:.1} bits), rms={:.3e} ({:.1} bits)",
            std::any::type_name::<BE>(),
            routes,
            max_err,
            max_bits,
            rms,
            rms_bits,
        );

        assert!(
            max_bits >= PHASE3B_MIN_PREC_BITS,
            "ship_sheared_phase3b_one_digit_one_output: max precision {:.1} bits < {:.1} bits (max_err={:.3e}, routes={:?})",
            max_bits,
            PHASE3B_MIN_PREC_BITS,
            max_err,
            routes,
        );
    }

    // ---------------------------------------------------------------------
    // Phase 3C: fuse the four Phase-3B source/key terms in the DFT domain and
    // perform only one final IDFT + normalization.
    // ---------------------------------------------------------------------
    {
        const PHASE3C_BASE: usize = 4;
        const PHASE3C_WEIGHT: usize = 5;
        const PHASE3C_OUT_GROUP: usize = 7;
        const PHASE3C_MIN_PREC_BITS: f64 = 18.0;

        let branch_re: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((17 * r + 13 * j + 7) % 101) as f64;
                        F::from_f64((raw - 50.0) / 64.0).unwrap()
                    })
                    .collect()
            })
            .collect();
        let branch_im: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((11 * r + 19 * j + 3) % 97) as f64;
                        F::from_f64((raw - 48.0) / 96.0).unwrap()
                    })
                    .collect()
            })
            .collect();

        let offsets: Vec<usize> = (0..(GROUPS - 1)).collect();
        let mut want_re = vec![F::zero(); m];
        let mut want_im = vec![F::zero(); m];
        let mut selector_hits = vec![0usize; m];
        let mut routes = Vec::with_capacity(PHASE3C_BASE);
        let mut sources: Vec<crate::layouts::CKKSCiphertextOwned<BE>> = Vec::with_capacity(PHASE3C_BASE);
        let mut keys = Vec::with_capacity(PHASE3C_BASE);

        for candidate in 0..PHASE3C_BASE {
            let runtime_rot = candidate * PHASE3C_WEIGHT;
            let route = crate::default::ship::mux::ship_sheared_runtime_route_from_output(PHASE3C_OUT_GROUP, runtime_rot, GROUPS);
            routes.push((route.group, route.physical_rot));

            let src_re: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_re[lane][(p + route.group) % m]
                })
                .collect();
            let src_im: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_im[lane][(p + route.group) % m]
                })
                .collect();

            let beta_slots = ship_sheared_packed_beta(&offsets, PHASE3C_WEIGHT, PHASE3C_BASE, candidate, m);
            let beta_re: Vec<F> = beta_slots
                .iter()
                .map(|&bit| F::from_f64(if bit { 1.0 } else { 0.0 }).unwrap())
                .collect();
            let beta_im = vec![F::zero(); m];
            let mut beta_coeffs = vec![F::zero(); params.n];
            encoder.pack_reim_coeffs(&mut beta_coeffs, &beta_re, &beta_im).unwrap();
            let beta_hat: Vec<i64> = beta_coeffs
                .iter()
                .map(|x| (x.to_f64().unwrap() * beta_scale).round() as i64)
                .collect();

            for p in 0..m {
                if beta_slots[p] {
                    selector_hits[p] += 1;
                    let lane = p % GROUPS;
                    let logical = (p + PHASE3C_OUT_GROUP + m - (runtime_rot % m)) % m;
                    want_re[p] = branch_re[lane][logical];
                    want_im[p] = branch_im[lane][logical];
                }
            }

            sources.push(ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                k,
                &src_re,
                &src_im,
                &mut scratch.borrow(),
            ));

            let mut key_xe = Source::new([80u8 + candidate as u8; 32]);
            let mut key_xa = Source::new([90u8 + candidate as u8; 32]);
            let key = hmux_rot_packed_key_encrypt_sk(
                module,
                host_module,
                &sk_host,
                &beta_hat,
                route.physical_rot,
                k,
                params.base2k.into(),
                mux_dsize,
                &mut key_xe,
                &mut key_xa,
                &mut scratch.borrow(),
            )
            .unwrap();
            let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key.key());
            module.glwe_switching_key_prepare(&mut prepared, key.key(), &mut scratch.borrow());
            keys.push(HMuxRotKeyPrepared {
                key: prepared,
                gal_el: key.gal_el(),
            });
        }

        assert!(selector_hits.iter().all(|&hits| hits == 1));
        assert_eq!(routes, vec![(7usize, 0usize), (2, 0), (29, 32), (24, 32)]);

        let source_refs: Vec<&crate::layouts::CKKSCiphertextOwned<BE>> = sources.iter().collect();
        let plans = crate::default::ship::mux::ship_mux_plans(module, std::iter::once(keys.as_slice()));

        // Copy only to initialize matching CKKS metadata/storage. The fused
        // primitive overwrites both GLWE columns completely.
        let mut fused = alloc_ct(&params, module, k);
        module.ckks_copy(&mut fused, &sources[0], &mut scratch.borrow()).unwrap();

        // Sequential source DFTs reuse the same `a_dft`; therefore the existing
        // single-source scratch bound is also sufficient for the fused path.
        let fused_bytes = crate::default::ship::mux::ship_mux_rotate_tmp_bytes(module, &sources[0], &keys[0].key, keys.len());
        let mut fused_scratch = ScratchOwned::<BE>::alloc(fused_bytes);

        crate::default::ship::mux::ship_mux_rotate_multi_source_with_offset(
            module,
            &mut fused,
            &source_refs,
            &keys,
            &plans,
            -(PACKED_BETA_SCALE_BITS as i64),
            &mut fused_scratch.borrow(),
        )
        .unwrap();

        let (got_re, got_im) = ckks_decrypt_decode(&params, module, &encoder, &fused, &sk, &mut scratch.borrow());

        let mut max_err = 0.0f64;
        let mut sq_err = 0.0f64;
        for p in 0..m {
            let er = (got_re[p].to_f64().unwrap() - want_re[p].to_f64().unwrap()).abs();
            let ei = (got_im[p].to_f64().unwrap() - want_im[p].to_f64().unwrap()).abs();
            max_err = max_err.max(er).max(ei);
            sq_err += er * er + ei * ei;
        }
        let rms = (sq_err / (2 * m) as f64).sqrt();
        let max_bits = if max_err == 0.0 { f64::INFINITY } else { -max_err.log2() };
        let rms_bits = if rms == 0.0 { f64::INFINITY } else { -rms.log2() };

        println!(
            "[sheared phase3c fused] backend={} routes={:?} max_err={:.3e} ({:.1} bits), rms={:.3e} ({:.1} bits)",
            std::any::type_name::<BE>(),
            routes,
            max_err,
            max_bits,
            rms,
            rms_bits,
        );

        assert!(
            max_bits >= PHASE3C_MIN_PREC_BITS,
            "ship_sheared_phase3c_fused_one_digit_one_output: max precision {:.1} bits < {:.1} bits (max_err={:.3e}, routes={:?})",
            max_bits,
            PHASE3C_MIN_PREC_BITS,
            max_err,
            routes,
        );
    }

    // ---------------------------------------------------------------------
    // Phase 4A: one complete packed digit over the full 32-group sheared
    // state.  Every output group is evaluated with the fused multi-source
    // primitive validated in Phase 3C.
    // ---------------------------------------------------------------------
    {
        const PHASE4_BASE: usize = 4;
        const PHASE4_WEIGHT: usize = 5;
        const PHASE4_MIN_PREC_BITS: f64 = 18.0;

        // Synthetic logical branch table x_r[j].  Reuse the same construction
        // as Phases 3B/3C, but materialize every sheared source group Y_s.
        let branch_re: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((17 * r + 13 * j + 7) % 101) as f64;
                        F::from_f64((raw - 50.0) / 64.0).unwrap()
                    })
                    .collect()
            })
            .collect();
        let branch_im: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((11 * r + 19 * j + 3) % 97) as f64;
                        F::from_f64((raw - 48.0) / 96.0).unwrap()
                    })
                    .collect()
            })
            .collect();

        // Encode each of the four packed selectors only once.  Their key
        // material will still depend on the physical rotation (rho=0 or 32).
        let offsets: Vec<usize> = (0..(GROUPS - 1)).collect();
        let mut beta_slots_by_candidate = Vec::with_capacity(PHASE4_BASE);
        let mut beta_hat_by_candidate = Vec::with_capacity(PHASE4_BASE);

        for candidate in 0..PHASE4_BASE {
            let beta_slots = ship_sheared_packed_beta(&offsets, PHASE4_WEIGHT, PHASE4_BASE, candidate, m);
            let beta_re: Vec<F> = beta_slots
                .iter()
                .map(|&bit| F::from_f64(if bit { 1.0 } else { 0.0 }).unwrap())
                .collect();
            let beta_im = vec![F::zero(); m];
            let mut beta_coeffs = vec![F::zero(); params.n];
            encoder.pack_reim_coeffs(&mut beta_coeffs, &beta_re, &beta_im).unwrap();
            let beta_hat: Vec<i64> = beta_coeffs
                .iter()
                .map(|x| (x.to_f64().unwrap() * beta_scale).round() as i64)
                .collect();

            beta_slots_by_candidate.push(beta_slots);
            beta_hat_by_candidate.push(beta_hat);
        }

        // The four selectors must partition every residue lane exactly once.
        for p in 0..m {
            let hits = beta_slots_by_candidate.iter().filter(|beta| beta[p]).count();
            assert_eq!(hits, 1, "Phase 4A selector partition failed at packed slot {p}");
        }

        // Materialize all 32 sheared input groups:
        //
        //     Y_s[p] = x_{p mod G}[(p+s) mod m].
        let mut sheared_sources: Vec<crate::layouts::CKKSCiphertextOwned<BE>> = Vec::with_capacity(GROUPS);
        for source_group in 0..GROUPS {
            let src_re: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_re[lane][(p + source_group) % m]
                })
                .collect();
            let src_im: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_im[lane][(p + source_group) % m]
                })
                .collect();

            sheared_sources.push(ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                k,
                &src_re,
                &src_im,
                &mut scratch.borrow(),
            ));
        }

        let mut worst_bits = f64::INFINITY;
        let mut worst_group = 0usize;
        let mut worst_err = 0.0f64;
        let mut total_sq_err = 0.0f64;
        let mut total_components = 0usize;

        for out_group in 0..GROUPS {
            let mut routes = Vec::with_capacity(PHASE4_BASE);
            let mut keys = Vec::with_capacity(PHASE4_BASE);
            let mut source_indices = Vec::with_capacity(PHASE4_BASE);

            let mut want_re = vec![F::zero(); m];
            let mut want_im = vec![F::zero(); m];

            for candidate in 0..PHASE4_BASE {
                let runtime_rot = candidate * PHASE4_WEIGHT;
                let route = crate::default::ship::mux::ship_sheared_runtime_route_from_output(out_group, runtime_rot, GROUPS);

                assert_eq!(
                    route.physical_rot % GROUPS,
                    0,
                    "Phase 4A physical rotation left the fixed residue lane"
                );

                routes.push((route.group, route.physical_rot));
                source_indices.push(route.group);

                let beta_slots = &beta_slots_by_candidate[candidate];
                for p in 0..m {
                    if beta_slots[p] {
                        let lane = p % GROUPS;
                        let logical = (p + out_group + m - (runtime_rot % m)) % m;
                        want_re[p] = branch_re[lane][logical];
                        want_im[p] = branch_im[lane][logical];
                    }
                }

                let seed_id = (out_group * PHASE4_BASE + candidate) as u8;
                let mut key_xe = Source::new([10u8.wrapping_add(seed_id); 32]);
                let mut key_xa = Source::new([150u8.wrapping_add(seed_id); 32]);

                let key = hmux_rot_packed_key_encrypt_sk(
                    module,
                    host_module,
                    &sk_host,
                    &beta_hat_by_candidate[candidate],
                    route.physical_rot,
                    k,
                    params.base2k.into(),
                    mux_dsize,
                    &mut key_xe,
                    &mut key_xa,
                    &mut scratch.borrow(),
                )
                .unwrap();

                let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key.key());
                module.glwe_switching_key_prepare(&mut prepared, key.key(), &mut scratch.borrow());

                keys.push(HMuxRotKeyPrepared {
                    key: prepared,
                    gal_el: key.gal_el(),
                });
            }

            let source_refs: Vec<&crate::layouts::CKKSCiphertextOwned<BE>> =
                source_indices.iter().map(|&idx| &sheared_sources[idx]).collect();

            let plans = crate::default::ship::mux::ship_mux_plans(module, std::iter::once(keys.as_slice()));

            let mut out = alloc_ct(&params, module, k);
            module.ckks_copy(&mut out, source_refs[0], &mut scratch.borrow()).unwrap();

            let mux_bytes =
                crate::default::ship::mux::ship_mux_rotate_tmp_bytes(module, source_refs[0], &keys[0].key, keys.len());
            let mut mux_scratch = ScratchOwned::<BE>::alloc(mux_bytes);

            crate::default::ship::mux::ship_mux_rotate_multi_source_with_offset(
                module,
                &mut out,
                &source_refs,
                &keys,
                &plans,
                -(PACKED_BETA_SCALE_BITS as i64),
                &mut mux_scratch.borrow(),
            )
            .unwrap();

            let (got_re, got_im) = ckks_decrypt_decode(&params, module, &encoder, &out, &sk, &mut scratch.borrow());

            let mut group_max_err = 0.0f64;
            for p in 0..m {
                let er = (got_re[p].to_f64().unwrap() - want_re[p].to_f64().unwrap()).abs();
                let ei = (got_im[p].to_f64().unwrap() - want_im[p].to_f64().unwrap()).abs();

                group_max_err = group_max_err.max(er).max(ei);
                total_sq_err += er * er + ei * ei;
                total_components += 2;
            }

            let group_bits = if group_max_err == 0.0 {
                f64::INFINITY
            } else {
                -group_max_err.log2()
            };

            if group_bits < worst_bits {
                worst_bits = group_bits;
                worst_group = out_group;
                worst_err = group_max_err;
            }

            assert!(
                group_bits >= PHASE4_MIN_PREC_BITS,
                "ship_sheared_phase4a_full_digit_state: output group {} precision {:.1} bits < {:.1} bits (max_err={:.3e}, routes={:?})",
                out_group,
                group_bits,
                PHASE4_MIN_PREC_BITS,
                group_max_err,
                routes,
            );
        }

        let global_rms = (total_sq_err / total_components as f64).sqrt();
        let global_rms_bits = if global_rms == 0.0 {
            f64::INFINITY
        } else {
            -global_rms.log2()
        };

        println!(
            "[sheared phase4a full-state] backend={} groups={} worst_group={} worst_max_err={:.3e} ({:.1} bits), global_rms={:.3e} ({:.1} bits)",
            std::any::type_name::<BE>(),
            GROUPS,
            worst_group,
            worst_err,
            worst_bits,
            global_rms,
            global_rms_bits,
        );
    }

    // ---------------------------------------------------------------------
    // Phase 4C: full mixed-radix BRotMux over the complete 32-group sheared
    // state, now with prepared packed H-MUX keys cached and reused across all
    // output groups.
    //
    // A packed functional key is determined by:
    //      (digit, candidate, physical_rot)
    // and NOT by the output group itself. For this test schedule this reduces
    // 32*(4+4+2)=320 freshly generated keys to only 7+6+2=15.
    // ---------------------------------------------------------------------
    {
        const PHASE4C_THETA: usize = 4;
        const PHASE4C_DIGITS: [(usize, usize); 3] = [(4, 4), (16, 4), (64, 2)];
        const PHASE4C_EXPECTED_KEYS_PER_DIGIT: [usize; 3] = [7, 6, 2];
        const PHASE4C_MIN_PREC_BITS: f64 = 16.0;

        assert_eq!(m, PHASE4C_THETA * 4 * 4 * 2);
        assert_eq!(GROUPS, 32);

        let support_offsets: Vec<usize> = (0..(GROUPS - 1)).map(|r| (37 * r + 3) % m).collect();

        let branch_re: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((17 * r + 13 * j + 7) % 101) as f64;
                        F::from_f64((raw - 50.0) / 64.0).unwrap()
                    })
                    .collect()
            })
            .collect();
        let branch_im: Vec<Vec<F>> = (0..GROUPS)
            .map(|r| {
                (0..m)
                    .map(|j| {
                        let raw = ((11 * r + 19 * j + 3) % 97) as f64;
                        F::from_f64((raw - 48.0) / 96.0).unwrap()
                    })
                    .collect()
            })
            .collect();

        let mut state: Vec<crate::layouts::CKKSCiphertextOwned<BE>> = Vec::with_capacity(GROUPS);
        for group in 0..GROUPS {
            let src_re: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_re[lane][(p + group) % m]
                })
                .collect();
            let src_im: Vec<F> = (0..m)
                .map(|p| {
                    let lane = p % GROUPS;
                    branch_im[lane][(p + group) % m]
                })
                .collect();

            state.push(ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                k,
                &src_re,
                &src_im,
                &mut scratch.borrow(),
            ));
        }

        let mut total_cached_keys = 0usize;

        for (digit_index, &(weight, base)) in PHASE4C_DIGITS.iter().enumerate() {
            let mut beta_slots_by_candidate = Vec::with_capacity(base);
            let mut beta_hat_by_candidate = Vec::with_capacity(base);

            for candidate in 0..base {
                let beta_slots = ship_sheared_packed_beta(&support_offsets, weight, base, candidate, m);
                let beta_re: Vec<F> = beta_slots
                    .iter()
                    .map(|&bit| F::from_f64(if bit { 1.0 } else { 0.0 }).unwrap())
                    .collect();
                let beta_im = vec![F::zero(); m];
                let mut beta_coeffs = vec![F::zero(); params.n];
                encoder.pack_reim_coeffs(&mut beta_coeffs, &beta_re, &beta_im).unwrap();
                let beta_hat: Vec<i64> = beta_coeffs
                    .iter()
                    .map(|x| (x.to_f64().unwrap() * beta_scale).round() as i64)
                    .collect();

                beta_slots_by_candidate.push(beta_slots);
                beta_hat_by_candidate.push(beta_hat);
            }

            for p in 0..m {
                let hits = beta_slots_by_candidate.iter().filter(|beta| beta[p]).count();
                assert_eq!(
                    hits, 1,
                    "Phase 4C digit {digit_index}: selector partition failed at packed slot {p}"
                );
            }

            // rho_cache[d][i] is paired with key_cache[d][i].
            let mut rho_cache: Vec<Vec<usize>> = Vec::with_capacity(base);

            for candidate in 0..base {
                let runtime_rot = candidate * weight;
                let mut rhos = Vec::new();

                for out_group in 0..GROUPS {
                    let route = crate::default::ship::mux::ship_sheared_runtime_route_from_output(out_group, runtime_rot, GROUPS);
                    assert_eq!(route.physical_rot % GROUPS, 0);
                    if !rhos.contains(&route.physical_rot) {
                        rhos.push(route.physical_rot);
                    }
                }

                rhos.sort_unstable();
                rho_cache.push(rhos);
            }

            let mut key_cache = Vec::with_capacity(base);

            for candidate in 0..base {
                let mut candidate_keys = Vec::with_capacity(rho_cache[candidate].len());

                for &rho in &rho_cache[candidate] {
                    let key_id = digit_index * 10_000 + candidate * 1_000 + rho;

                    let mut xe_seed = [0u8; 32];
                    let mut xa_seed = [0u8; 32];
                    xe_seed[..8].copy_from_slice(&(key_id as u64).to_le_bytes());
                    xa_seed[..8].copy_from_slice(&(key_id as u64).to_le_bytes());
                    xe_seed[8] = 0x4c;
                    xa_seed[8] = 0xc4;

                    let mut key_xe = Source::new(xe_seed);
                    let mut key_xa = Source::new(xa_seed);

                    let key = hmux_rot_packed_key_encrypt_sk(
                        module,
                        host_module,
                        &sk_host,
                        &beta_hat_by_candidate[candidate],
                        rho,
                        k,
                        params.base2k.into(),
                        mux_dsize,
                        &mut key_xe,
                        &mut key_xa,
                        &mut scratch.borrow(),
                    )
                    .unwrap();

                    let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key.key());
                    module.glwe_switching_key_prepare(&mut prepared, key.key(), &mut scratch.borrow());

                    candidate_keys.push(HMuxRotKeyPrepared {
                        key: prepared,
                        gal_el: key.gal_el(),
                    });
                }

                key_cache.push(candidate_keys);
            }

            let cached_this_digit: usize = key_cache.iter().map(Vec::len).sum();
            assert_eq!(
                cached_this_digit, PHASE4C_EXPECTED_KEYS_PER_DIGIT[digit_index],
                "Phase 4C digit {digit_index}: unexpected unique key count; rhos={rho_cache:?}"
            );
            total_cached_keys += cached_this_digit;

            let plans = crate::default::ship::mux::ship_mux_plans(module, key_cache.iter().map(Vec::as_slice));

            let mut next_state = Vec::with_capacity(GROUPS);
            let mut max_rho = 0usize;

            for out_group in 0..GROUPS {
                let mut source_indices = Vec::with_capacity(base);
                let mut key_refs = Vec::with_capacity(base);

                for candidate in 0..base {
                    let runtime_rot = candidate * weight;
                    let route = crate::default::ship::mux::ship_sheared_runtime_route_from_output(out_group, runtime_rot, GROUPS);

                    assert_eq!(
                        route.physical_rot % GROUPS,
                        0,
                        "Phase 4C digit {digit_index}: physical rotation left the fixed lane"
                    );
                    max_rho = max_rho.max(route.physical_rot);
                    source_indices.push(route.group);

                    let cache_index = rho_cache[candidate]
                        .iter()
                        .position(|&rho| rho == route.physical_rot)
                        .expect("Phase 4C route rho must exist in the key cache");

                    key_refs.push(&key_cache[candidate][cache_index]);
                }

                let source_refs: Vec<&crate::layouts::CKKSCiphertextOwned<BE>> =
                    source_indices.iter().map(|&idx| &state[idx]).collect();

                let mut out = alloc_ct(&params, module, k);
                module.ckks_copy(&mut out, source_refs[0], &mut scratch.borrow()).unwrap();

                let mux_bytes = crate::default::ship::mux::ship_mux_rotate_tmp_bytes(
                    module,
                    source_refs[0],
                    &key_refs[0].key,
                    key_refs.len(),
                );
                let mut mux_scratch = ScratchOwned::<BE>::alloc(mux_bytes);

                crate::default::ship::mux::ship_mux_rotate_multi_source_refs_with_offset(
                    module,
                    &mut out,
                    &source_refs,
                    &key_refs,
                    &plans,
                    -(PACKED_BETA_SCALE_BITS as i64),
                    &mut mux_scratch.borrow(),
                )
                .unwrap();

                next_state.push(out);
            }

            state = next_state;

            println!(
                "[sheared phase4c cache] backend={} digit={} weight={} base={} unique_keys={} rhos={:?} max_physical_rot={}",
                std::any::type_name::<BE>(),
                digit_index,
                weight,
                base,
                cached_this_digit,
                rho_cache,
                max_rho,
            );
        }

        assert_eq!(total_cached_keys, 15, "Phase 4C expected exactly 15 unique packed H-MUX keys");

        let mut lane_shift = vec![0usize; GROUPS];
        for lane in 0..GROUPS - 1 {
            let u = support_offsets[lane];
            let mut shift = 0usize;
            for &(weight, base) in &PHASE4C_DIGITS {
                shift += ((u / weight) % base) * weight;
            }
            assert_eq!(
                shift,
                u - (u % PHASE4C_THETA),
                "Phase 4C mixed-radix reconstruction failed for lane {lane}"
            );
            lane_shift[lane] = shift % m;
        }

        let mut worst_bits = f64::INFINITY;
        let mut worst_group = 0usize;
        let mut worst_slot = 0usize;
        let mut worst_err = 0.0f64;
        let mut total_sq_err = 0.0f64;
        let mut total_components = 0usize;

        for group in 0..GROUPS {
            let (got_re, got_im) = ckks_decrypt_decode(&params, module, &encoder, &state[group], &sk, &mut scratch.borrow());

            let mut group_max_err = 0.0f64;
            let mut group_worst_slot = 0usize;

            for p in 0..m {
                let lane = p % GROUPS;
                let logical = (p + group + m - lane_shift[lane]) % m;
                let want_re = branch_re[lane][logical].to_f64().unwrap();
                let want_im = branch_im[lane][logical].to_f64().unwrap();

                let er = (got_re[p].to_f64().unwrap() - want_re).abs();
                let ei = (got_im[p].to_f64().unwrap() - want_im).abs();
                let local = er.max(ei);

                if local > group_max_err {
                    group_max_err = local;
                    group_worst_slot = p;
                }

                total_sq_err += er * er + ei * ei;
                total_components += 2;
            }

            let group_bits = if group_max_err == 0.0 {
                f64::INFINITY
            } else {
                -group_max_err.log2()
            };

            if group_bits < worst_bits {
                worst_bits = group_bits;
                worst_group = group;
                worst_slot = group_worst_slot;
                worst_err = group_max_err;
            }

            assert!(
                group_bits >= PHASE4C_MIN_PREC_BITS,
                "ship_sheared_phase4c_cached_full_brotmux: group {} precision {:.1} bits < {:.1} bits (max_err={:.3e}, slot={})",
                group,
                group_bits,
                PHASE4C_MIN_PREC_BITS,
                group_max_err,
                group_worst_slot,
            );
        }

        let global_rms = (total_sq_err / total_components as f64).sqrt();
        let global_rms_bits = if global_rms == 0.0 {
            f64::INFINITY
        } else {
            -global_rms.log2()
        };

        let naive_keys = GROUPS * PHASE4C_DIGITS.iter().map(|&(_, b)| b).sum::<usize>();

        println!(
            "[sheared phase4c full-brotmux] backend={} groups={} digits={:?} cached_keys={} naive_keys={} reduction={:.2}x worst_group={} worst_slot={} worst_max_err={:.3e} ({:.1} bits), global_rms={:.3e} ({:.1} bits)",
            std::any::type_name::<BE>(),
            GROUPS,
            PHASE4C_DIGITS,
            total_cached_keys,
            naive_keys,
            naive_keys as f64 / total_cached_keys as f64,
            worst_group,
            worst_slot,
            worst_err,
            worst_bits,
            global_rms,
            global_rms_bits,
        );
    }

    // Phase 5A: real SHIP coefficient-encoding + mask schedule -> sheared
    // BRotMCol state. The ordinary Poulpy BRotMCol ciphertexts are the golden
    // reference; the host-side packed construction is only a representation
    // transform of those ordinary intermediate nodes.
    {
        use poulpy_core::GLWEKeyswitch as _;
        use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef as _;

        const PHASE5A_MIN_BITS_FFT: f64 = 8.0;
        const PHASE5A_MIN_BITS_EXACT: f64 = 14.0;

        let phase5a_plan = ship_sheared_suite_plan(&params);
        let phase5a_h = phase5a_plan.sparse_hamming_weight();
        let phase5a_groups = (phase5a_h + 1).next_power_of_two();
        let phase5a_public_lane = phase5a_h;
        let phase5a_kk = phase5a_plan.raised_k(params.base2k);
        let phase5a_ld = phase5a_plan.log_delta_work();

        assert!(phase5a_groups <= m && m % phase5a_groups == 0);
        assert_eq!(phase5a_h, 31);
        assert_eq!(phase5a_groups, 32);

        let phase5a_work = CKKSTestParams {
            k: phase5a_kk,
            prec_meta: crate::CKKSMeta {
                log_delta: phase5a_ld,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            },
            prec_log_budget: phase5a_kk - phase5a_ld,
            ..params
        };
        let (phase5a_sk_host, _, phase5a_sk) = gen_sk_with_host(&phase5a_work, module, host_module, [0xa5u8; 32]);

        let gamma = 2f64.powi(phase5a_plan.log_gamma() as i32);
        let phase5a_coeffs: Vec<F> = (0..params.n)
            .map(|i| {
                let v = if i < m {
                    (((29 * i + 7) % 97) as f64 - 48.0) / 128.0
                } else {
                    0.0
                };
                F::from_f64(v / gamma).unwrap()
            })
            .collect();

        let bottom_prec = ckks_spec(params.n, params.base2k, params.base2k, 0);
        let mut host_pt = host_module.ckks_pt_vec_alloc(params.base2k.into(), bottom_prec.k());
        host_pt.set_meta(bottom_prec.meta());
        host_pt.encode_host_floats(&phase5a_coeffs).unwrap();

        let phase5a_ct0 = ckks_encrypt_pt(
            &phase5a_work,
            module,
            &phase5a_sk,
            params.base2k,
            &host_pt,
            &mut scratch.borrow(),
        );

        let mut phase5a_source = Source::new([0x5au8; 32]);
        let phase5a_spec = ShipSecretSpec::sample(&phase5a_plan, &mut phase5a_source);

        let phase5a_layout = ShipKeysLayout {
            mux_dsize,
            tensor_key: phase5a_work.tsk_layout().layout,
            conjugation_key: phase5a_work.atk_layout().layout,
            complex: false,
        };
        let mut phase5a_key_xe = Source::new([0x61u8; 32]);
        let mut phase5a_key_xa = Source::new([0x62u8; 32]);
        let mut phase5a_key_scratch = alloc_scratch(&phase5a_work, module);
        let phase5a_key_set = ShipKeySet::generate::<BE, F>(
            module,
            host_module,
            &phase5a_plan,
            params.base2k.into(),
            &phase5a_spec,
            &phase5a_sk_host,
            &phase5a_layout,
            &mut phase5a_key_xe,
            &mut phase5a_key_xa,
            &mut phase5a_key_scratch.borrow(),
        )
        .unwrap();
        let phase5a_keys = phase5a_key_set.prepare(module, &mut phase5a_key_scratch.borrow()).unwrap();

        // Match production SHIP: dense -> sparse first, coefficient encodings second.
        let mut phase5a_sparse = module.ckks_ciphertext_alloc(params.base2k.into(), params.base2k.into());
        module.glwe_keyswitch(
            &mut phase5a_sparse,
            &phase5a_ct0,
            &phase5a_keys.dense_to_sparse().to_backend_ref(),
            &mut scratch.borrow(),
        );
        phase5a_sparse.set_meta_checked(phase5a_ct0.meta()).unwrap();

        let phase5a_enc = BE::ckks_ship_coeff_encodings_impl::<F, _>(
            module,
            &phase5a_sparse,
            &phase5a_plan,
            params.base2k.into(),
            false,
            &mut scratch.borrow(),
        )
        .unwrap();
        let phase5a_clear = crate::encoding::ship::coeff_enc::ship_coeff_slot_values_host::<_, F>(
            &phase5a_sparse,
            &phase5a_plan,
            params.base2k.into(),
            false,
        )
        .unwrap();
        let (pt0_re, pt0_im) = phase5a_clear.pt0;
        let pi_slots = phase5a_clear.pi;
        let terms = 4 * phase5a_plan.theta();

        // Ordinary production BRotMCol intermediate ciphertexts.
        let phase5a_masking_bytes = crate::default::ship::masking::ship_masking_tmp_bytes(module, &phase5a_plan, params.base2k);
        let mut ordinary_slots: Vec<(Vec<F>, Vec<F>)> = Vec::with_capacity(phase5a_groups);
        for (lane, ik) in phase5a_keys.index_keys().iter().enumerate() {
            let mut ordinary = module.ckks_ciphertext_alloc(params.base2k.into(), phase5a_kk.into());
            let mut ordinary_scratch = ScratchOwned::<BE>::alloc(phase5a_masking_bytes);
            crate::default::ship::masking::ship_masking_accumulate(
                module,
                &mut ordinary,
                &phase5a_plan,
                ik.masks(),
                &phase5a_enc.pi[lane],
                [0, 1, 2, 3],
                &mut ordinary_scratch.borrow(),
            )
            .unwrap();
            ordinary_slots.push(ckks_decrypt_decode::<BE, F, E>(
                &phase5a_work,
                module,
                &encoder,
                &ordinary,
                &phase5a_sk,
                &mut scratch.borrow(),
            ));
        }

        let mut ordinary_pt0 = module.ckks_ciphertext_alloc(params.base2k.into(), phase5a_kk.into());
        module.glwe_zero(&mut ordinary_pt0);
        ordinary_pt0
            .set_meta_checked(crate::CKKSMeta {
                log_delta: phase5a_ld,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            })
            .unwrap();
        module
            .ckks_add_pt_vec_assign(&mut ordinary_pt0, &phase5a_enc.pt0, &mut scratch.borrow())
            .unwrap();
        ordinary_slots.push(ckks_decrypt_decode::<BE, F, E>(
            &phase5a_work,
            module,
            &encoder,
            &ordinary_pt0,
            &phase5a_sk,
            &mut scratch.borrow(),
        ));
        assert_eq!(ordinary_slots.len(), phase5a_groups);

        // Host packed/sheared construction. The masks/pi here are the data being
        // repacked; expected values come only from the ordinary ciphertexts above.
        let mut masks_by_lane: Vec<Vec<Vec<F>>> = Vec::with_capacity(phase5a_h);
        for (lane, &(j, s_j)) in phase5a_spec.support().iter().enumerate() {
            let u = phase5a_spec.offset(&phase5a_plan, lane);
            masks_by_lane.push(crate::encoding::ship::masks::ship_mask_slot_vectors::<F>(
                &phase5a_plan,
                lane,
                j,
                s_j,
                u,
                false,
            ));
        }

        let mut max_err = 0.0f64;
        let mut worst_group = 0usize;
        let mut worst_slot = 0usize;
        for group in 0..phase5a_groups {
            let mut packed_re = vec![F::zero(); m];
            let mut packed_im = vec![F::zero(); m];

            for term in 0..terms {
                for p in 0..m {
                    let lane = p % phase5a_groups;
                    if lane >= phase5a_h {
                        continue;
                    }
                    let q = (p + group) % m;
                    packed_re[p] = packed_re[p] + masks_by_lane[lane][term][q] * pi_slots[lane][term].0[q];
                    packed_im[p] = packed_im[p] + masks_by_lane[lane][term][q] * pi_slots[lane][term].1[q];
                }
            }

            for p in 0..m {
                let lane = p % phase5a_groups;
                let q = (p + group) % m;
                if lane == phase5a_public_lane {
                    packed_re[p] = pt0_re[q];
                    packed_im[p] = pt0_im[q];
                }

                let want_re = ordinary_slots[lane].0[q];
                let want_im = ordinary_slots[lane].1[q];
                let er = (packed_re[p].to_f64().unwrap() - want_re.to_f64().unwrap()).abs();
                let ei = (packed_im[p].to_f64().unwrap() - want_im.to_f64().unwrap()).abs();
                let local = er.max(ei);
                if local > max_err {
                    max_err = local;
                    worst_group = group;
                    worst_slot = p;
                }
            }
        }

        let bits = if max_err == 0.0 { f64::INFINITY } else { -max_err.log2() };
        let required_bits = if phase5a_ld < 40 {
            PHASE5A_MIN_BITS_FFT
        } else {
            PHASE5A_MIN_BITS_EXACT
        };
        println!(
            "[golden-diff phase5a-brotmcol-layout] backend={} max_err={:.3e} ({:.2} bits) worst_group={} worst_slot={}",
            std::any::type_name::<BE>(),
            max_err,
            bits,
            worst_group,
            worst_slot,
        );
        assert!(
            bits >= required_bits,
            "golden-diff phase5a-brotmcol-layout failed: {:.2} bits < {:.2} bits (max_err={:.3e}, group={}, slot={})",
            bits,
            required_bits,
            max_err,
            worst_group,
            worst_slot,
        );
    }

    // ---------------------------------------------------------------------
    // Phase 5B-1: real encrypted packed BRotMCol for two padded sheared
    // groups. This is the first test that moves the Phase-5A bridge through
    // the actual CKKS encode -> encrypt -> prepare-left -> convolution path.
    //
    // We test g=0 and g=G-1. The latter forces the global (p+g) mod m wrap.
    // The public pt0 lane and multiplicative-identity padding are injected as
    // one plaintext after the encrypted BRotMCol accumulation.
    // ---------------------------------------------------------------------
    {
        const PHASE5B_MIN_BITS_FFT: f64 = 8.0;
        const PHASE5B_MIN_BITS_EXACT: f64 = 14.0;

        let plan = ship_sheared_suite_plan(&params);
        let h = plan.sparse_hamming_weight();
        let groups = (h + 1).next_power_of_two();
        let public_lane = h;
        let base2k = params.base2k;
        let kk = plan.raised_k(base2k);
        let ld = plan.log_delta_work();
        let k_pi = ld + base2k;
        let terms = 4 * plan.theta();

        assert!(groups <= m && m % groups == 0);
        assert_eq!(h, 31);
        assert_eq!(groups, 32);

        let phase5b_params = CKKSTestParams {
            k: kk,
            prec_meta: crate::CKKSMeta {
                log_delta: ld,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            },
            prec_log_budget: kk - ld,
            ..params
        };

        let (phase5b_sk_host, _, phase5b_sk) = gen_sk_with_host(&phase5b_params, module, host_module, [0xb5u8; 32]);

        let gamma = 2f64.powi(plan.log_gamma() as i32);
        let coeffs: Vec<F> = (0..params.n)
            .map(|i| {
                let v = if i < m {
                    (((29 * i + 7) % 97) as f64 - 48.0) / 128.0
                } else {
                    0.0
                };
                F::from_f64(v / gamma).unwrap()
            })
            .collect();

        let bottom_prec = ckks_spec(params.n, base2k, base2k, 0);
        let mut host_pt = host_module.ckks_pt_vec_alloc(base2k.into(), bottom_prec.k());
        host_pt.set_meta(bottom_prec.meta());
        host_pt.encode_host_floats(&coeffs).unwrap();

        let ct0 = ckks_encrypt_pt(&phase5b_params, module, &phase5b_sk, base2k, &host_pt, &mut scratch.borrow());

        // GOLDEN-DIFF: establish the exact production SHIP key/spec path first.
        // Both the ordinary golden path and the packed path below consume the
        // same sparse-encapsulated ciphertext and the same coefficient encodings.
        use poulpy_core::GLWEKeyswitch as _;
        use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef as _;

        let mut source = Source::new([0x5au8; 32]);
        let spec = ShipSecretSpec::sample(&plan, &mut source);
        assert_eq!(spec.support().len(), h);

        let golden_layout = ShipKeysLayout {
            mux_dsize,
            tensor_key: phase5b_params.tsk_layout().layout,
            conjugation_key: phase5b_params.atk_layout().layout,
            complex: false,
        };
        let mut golden_xe = Source::new([0x71u8; 32]);
        let mut golden_xa = Source::new([0x72u8; 32]);
        let mut golden_key_scratch = alloc_scratch(&phase5b_params, module);
        let golden_key_set = ShipKeySet::generate::<BE, F>(
            module,
            host_module,
            &plan,
            base2k.into(),
            &spec,
            &phase5b_sk_host,
            &golden_layout,
            &mut golden_xe,
            &mut golden_xa,
            &mut golden_key_scratch.borrow(),
        )
        .unwrap();
        let golden_keys = golden_key_set.prepare(module, &mut golden_key_scratch.borrow()).unwrap();

        // Match production ship_bootstrap_roots exactly: dense -> sparse first,
        // then derive pt0/pi from the public components of the sparse ciphertext.
        let mut a_sparse = module.ckks_ciphertext_alloc(base2k.into(), base2k.into());
        module.glwe_keyswitch(
            &mut a_sparse,
            &ct0,
            &golden_keys.dense_to_sparse().to_backend_ref(),
            &mut scratch.borrow(),
        );
        a_sparse.set_meta_checked(ct0.meta()).unwrap();

        let enc =
            BE::ckks_ship_coeff_encodings_impl::<F, _>(module, &a_sparse, &plan, base2k.into(), false, &mut scratch.borrow())
                .unwrap();
        assert_eq!(enc.pi.len(), h);

        let clear = crate::encoding::ship::coeff_enc::ship_coeff_slot_values_host::<_, F>(&a_sparse, &plan, base2k.into(), false)
            .unwrap();
        let (pt0_re, pt0_im) = clear.pt0;
        let pi_slots = clear.pi;
        assert!(pi_slots.iter().all(|v| v.len() == terms));

        let mut masks_by_lane: Vec<Vec<Vec<F>>> = Vec::with_capacity(h);
        for (lane, &(j, s_j)) in spec.support().iter().enumerate() {
            let u = spec.offset(&plan, lane);
            let masks = crate::encoding::ship::masks::ship_mask_slot_vectors::<F>(&plan, lane, j, s_j, u, false);
            assert_eq!(masks.len(), terms);
            masks_by_lane.push(masks);
        }

        let encode_slots = |k_pt: usize, re: &[F], im: &[F]| {
            assert_eq!(re.len(), m);
            assert_eq!(im.len(), m);

            let mut values = vec![F::zero(); 2 * m];
            values[..m].copy_from_slice(re);
            values[m..].copy_from_slice(im);

            let mut buffer = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&values);
            let mut pt = module.ckks_pt_vec_alloc(base2k.into(), k_pt.into());
            pt.set_meta_checked(crate::CKKSMeta {
                log_delta: ld,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            })
            .unwrap();
            module.ckks_encode_slots_assign_into(&mut pt, &mut buffer).unwrap();
            pt
        };

        let mask_size = kk.div_ceil(base2k);
        let mask_msb = poulpy_core::msb_mask_bottom_limb(base2k, kk);
        let prep_bytes = module.cnv_prepare_left_tmp_bytes(mask_size, mask_size);
        let masking_bytes = crate::default::ship::masking::ship_masking_tmp_bytes(module, &plan, base2k);

        let enc_infos = phase5b_params.glwe_layout();
        let mut mask_xe = Source::new([0xc1u8; 32]);
        let mut mask_xa = Source::new([0x1cu8; 32]);

        // GOLDEN-DIFF: ordinary Poulpy BRotMCol leaves in packed lane order
        // [branch0, ..., branch(h-1), pt0].  These use the production masks and
        // production scalar ship_masking_accumulate primitive.
        let golden_mux_plans = crate::default::ship::mux::ship_mux_plans(
            module,
            golden_keys
                .index_keys()
                .iter()
                .flat_map(|ik| ik.mux_keys().iter().map(Vec::as_slice)),
        );

        let mut golden_state: Vec<crate::layouts::CKKSCiphertextOwned<BE>> = Vec::with_capacity(groups);
        for (lane, ik) in golden_keys.index_keys().iter().enumerate() {
            let mut acc = module.ckks_ciphertext_alloc(base2k.into(), kk.into());
            let mut ordinary_masking_scratch = ScratchOwned::<BE>::alloc(masking_bytes);
            crate::default::ship::masking::ship_masking_accumulate(
                module,
                &mut acc,
                &plan,
                ik.masks(),
                &enc.pi[lane],
                [0, 1, 2, 3],
                &mut ordinary_masking_scratch.borrow(),
            )
            .unwrap();
            golden_state.push(acc);
        }

        let mut golden_pt0 = module.ckks_ciphertext_alloc(base2k.into(), kk.into());
        module.glwe_zero(&mut golden_pt0);
        golden_pt0
            .set_meta_checked(crate::CKKSMeta {
                log_delta: ld,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            })
            .unwrap();
        module
            .ckks_add_pt_vec_assign(&mut golden_pt0, &enc.pt0, &mut scratch.borrow())
            .unwrap();
        golden_state.push(golden_pt0);
        assert_eq!(golden_state.len(), groups);

        // Ordinary -> sheared representation conversion only.  No secret offset,
        // lane_shift, or hand-written SHIP formula is allowed in this gate:
        //     Y_g[p] = x_{p mod G}[(p + g) mod m].
        macro_rules! assert_packed_matches_ordinary_full_groups {
            ($tag:expr, $packed:expr, $ordinary:expr, $required_bits:expr) => {{
                let tag = $tag;
                assert_eq!(($packed).len(), groups);
                assert_eq!(($ordinary).len(), groups);

                let ordinary_slots: Vec<(Vec<F>, Vec<F>)> = ($ordinary)
                    .iter()
                    .map(|ct| {
                        ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            ct,
                            &phase5b_sk,
                            &mut scratch.borrow(),
                        )
                    })
                    .collect();

                let mut worst = 0.0f64;
                let mut worst_group = 0usize;
                let mut worst_slot = 0usize;
                let mut sum_sq = 0.0f64;
                let mut count = 0usize;

                for group in 0..groups {
                    let (have_re, have_im) = ckks_decrypt_decode::<BE, F, E>(
                        &phase5b_params,
                        module,
                        &encoder,
                        &($packed)[group],
                        &phase5b_sk,
                        &mut scratch.borrow(),
                    );
                    for p in 0..m {
                        let lane = p % groups;
                        let q = (p + group) % m;
                        let want_re = ordinary_slots[lane].0[q];
                        let want_im = ordinary_slots[lane].1[q];
                        let er = (have_re[p].to_f64().unwrap() - want_re.to_f64().unwrap()).abs();
                        let ei = (have_im[p].to_f64().unwrap() - want_im.to_f64().unwrap()).abs();
                        let local = er.max(ei);
                        if local > worst {
                            worst = local;
                            worst_group = group;
                            worst_slot = p;
                        }
                        sum_sq += er * er + ei * ei;
                        count += 2;
                    }
                }

                let bits = if worst == 0.0 { f64::INFINITY } else { -worst.log2() };
                let rms = (sum_sq / count as f64).sqrt();
                println!(
                    "[golden-diff {}] backend={} max_err={:.3e} ({:.2} bits) rms={:.3e} worst_group={} worst_slot={}",
                    tag,
                    std::any::type_name::<BE>(),
                    worst,
                    bits,
                    rms,
                    worst_group,
                    worst_slot,
                );
                assert!(
                    bits >= $required_bits,
                    "golden-diff {} failed: {:.2} bits < {:.2} bits (max_err={:.3e}, group={}, slot={})",
                    tag,
                    bits,
                    $required_bits,
                    worst,
                    worst_group,
                    worst_slot,
                );
            }};
        }

        let mut state: Vec<crate::layouts::CKKSCiphertextOwned<BE>> = Vec::with_capacity(groups);

        for group in 0..groups {
            let mut prepared_masks = Vec::with_capacity(terms);
            let mut packed_pis = Vec::with_capacity(terms);

            for term in 0..terms {
                let mut mask_re = vec![F::zero(); m];
                let mask_im = vec![F::zero(); m];
                let mut pi_re = vec![F::zero(); m];
                let mut pi_im = vec![F::zero(); m];

                for p in 0..m {
                    let lane = p % groups;
                    if lane >= h {
                        continue;
                    }
                    let q = (p + group) % m;

                    mask_re[p] = masks_by_lane[lane][term][q];
                    pi_re[p] = pi_slots[lane][term].0[q];
                    pi_im[p] = pi_slots[lane][term].1[q];
                }

                let mask_pt = encode_slots(kk, &mask_re, &mask_im);
                let mut mask_ct = module.ckks_ciphertext_alloc(base2k.into(), kk.into());
                module
                    .ckks_encrypt_sk(
                        &mut mask_ct,
                        &mask_pt,
                        &phase5b_sk,
                        &enc_infos,
                        &mut mask_xe,
                        &mut mask_xa,
                        &mut scratch.borrow(),
                    )
                    .unwrap();

                let mut prep = module.cnv_pvec_left_alloc(2, mask_size);
                let mut prep_scratch = ScratchOwned::<BE>::alloc(prep_bytes);
                module.cnv_prepare_left(
                    &mut prep.to_backend_mut(),
                    poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&mask_ct).data(),
                    mask_msb,
                    &mut prep_scratch.borrow(),
                );
                prepared_masks.push(prep);

                packed_pis.push(encode_slots(k_pi, &pi_re, &pi_im));
            }

            let mut acc = module.ckks_ciphertext_alloc(base2k.into(), kk.into());
            let mut masking_scratch = ScratchOwned::<BE>::alloc(masking_bytes);
            crate::default::ship::masking::ship_masking_accumulate(
                module,
                &mut acc,
                &plan,
                &prepared_masks,
                &packed_pis,
                [0, 1, 2, 3],
                &mut masking_scratch.borrow(),
            )
            .unwrap();

            let mut inject_re = vec![F::zero(); m];
            let mut inject_im = vec![F::zero(); m];

            for p in 0..m {
                let lane = p % groups;
                let q = (p + group) % m;
                if lane == public_lane {
                    inject_re[p] = pt0_re[q];
                    inject_im[p] = pt0_im[q];
                } else if lane >= h {
                    // Padding lanes, if any, are multiplicative identities.
                    inject_re[p] = F::from_f64(1.0).unwrap();
                }
            }

            // BRotMCol has already multiplied two ld-scale operands.

            // Match the injection plaintext budget to the post-BRotMCol

            // ciphertext budget instead of encoding it at the full raised width.

            assert_eq!(acc.log_delta(), ld, "Phase 5B-1 unexpected post-BRotMCol working scale");

            let inject_k = ld + acc.log_budget();

            let inject_pt = encode_slots(inject_k, &inject_re, &inject_im);

            assert_eq!(
                inject_pt.log_delta(),
                acc.log_delta(),
                "Phase 5B-1 injection plaintext scale mismatch"
            );
            assert_eq!(
                inject_pt.log_budget(),
                acc.log_budget(),
                "Phase 5B-1 injection plaintext budget mismatch"
            );

            module
                .ckks_add_pt_vec_assign(&mut acc, &inject_pt, &mut scratch.borrow())
                .unwrap();

            // Correctness is checked below only against ordinary Poulpy
            // BRotMCol ciphertexts after ordinary->sheared repacking.

            state.push(acc);
        }

        let phase5b1_required_bits = if ld < 40 {
            PHASE5B_MIN_BITS_FFT
        } else {
            PHASE5B_MIN_BITS_EXACT
        };
        assert_packed_matches_ordinary_full_groups!("phase5b1-brotmcol", &state, &golden_state, phase5b1_required_bits);
        // -----------------------------------------------------------------
        // Phase 5B-2: real h=31/G=32 packed BRotMux on top of the encrypted
        // packed BRotMCol state just constructed above.
        //
        // Unlike Phase 4C, the selector digits are derived from the real
        // ShipSecretSpec offsets and plan.mux_bases(); nothing here hardcodes
        // the synthetic (4,4),(16,4),(64,2) schedule.
        // -----------------------------------------------------------------
        {
            const PHASE5B2_MIN_BITS_FFT: f64 = 12.0;
            const PHASE5B2_MIN_BITS_EXACT: f64 = 18.0;

            assert_eq!(h, 31);
            assert_eq!(groups, 32);
            assert_eq!(public_lane, 31);
            assert_eq!(state.len(), groups);

            let support_offsets: Vec<usize> = (0..h).map(|lane| spec.offset(&plan, lane)).collect();

            let mut digits = Vec::new();
            let mut weight = plan.theta();
            for base in plan.mux_bases() {
                digits.push((weight, base));
                weight *= base;
            }
            assert!(!digits.is_empty());

            let packed_beta_scale = (1u64 << PACKED_BETA_SCALE_BITS) as f64;
            let mut total_cached_keys = 0usize;

            for (digit_index, &(digit_weight, base)) in digits.iter().enumerate() {
                // Golden side: execute the production scalar H-MUX digit on every
                // secret branch. The public pt0 lane is an identity branch.
                for lane in 0..h {
                    let key_group = &golden_keys.index_keys()[lane].mux_keys()[digit_index];
                    let mux_bytes = crate::default::ship::mux::ship_mux_rotate_tmp_bytes(
                        module,
                        &golden_state[lane],
                        &key_group[0].key,
                        key_group.len(),
                    );
                    let mut ordinary_mux_scratch = ScratchOwned::<BE>::alloc(mux_bytes);
                    crate::default::ship::mux::ship_mux_rotate(
                        module,
                        &mut golden_state[lane],
                        key_group,
                        &golden_mux_plans,
                        &mut ordinary_mux_scratch.borrow(),
                    )
                    .unwrap();
                }

                let mut beta_hat_by_candidate = Vec::with_capacity(base);
                let mut beta_slots_by_candidate = Vec::with_capacity(base);

                for candidate in 0..base {
                    let beta_slots = ship_sheared_packed_beta(&support_offsets, digit_weight, base, candidate, m);
                    let beta_re: Vec<F> = beta_slots
                        .iter()
                        .map(|&bit| F::from_f64(if bit { 1.0 } else { 0.0 }).unwrap())
                        .collect();
                    let beta_im = vec![F::zero(); m];
                    let mut beta_coeffs = vec![F::zero(); params.n];
                    encoder.pack_reim_coeffs(&mut beta_coeffs, &beta_re, &beta_im).unwrap();
                    let beta_hat: Vec<i64> = beta_coeffs
                        .iter()
                        .map(|x| (x.to_f64().unwrap() * packed_beta_scale).round() as i64)
                        .collect();

                    beta_slots_by_candidate.push(beta_slots);
                    beta_hat_by_candidate.push(beta_hat);
                }

                for p in 0..m {
                    let hits = beta_slots_by_candidate.iter().filter(|beta| beta[p]).count();
                    assert_eq!(
                        hits, 1,
                        "Phase 5B-2 digit {digit_index}: selectors do not partition packed slot {p}"
                    );
                }

                let mut rho_cache: Vec<Vec<usize>> = Vec::with_capacity(base);
                for candidate in 0..base {
                    let runtime_rot = candidate * digit_weight;
                    let mut rhos = Vec::new();
                    for out_group in 0..groups {
                        let route =
                            crate::default::ship::mux::ship_sheared_runtime_route_from_output(out_group, runtime_rot, groups);
                        assert_eq!(
                            route.physical_rot % groups,
                            0,
                            "Phase 5B-2 physical rotation left fixed residue lanes"
                        );
                        if !rhos.contains(&route.physical_rot) {
                            rhos.push(route.physical_rot);
                        }
                    }
                    rhos.sort_unstable();
                    rho_cache.push(rhos);
                }

                let mut key_cache = Vec::with_capacity(base);
                for candidate in 0..base {
                    let mut candidate_keys = Vec::with_capacity(rho_cache[candidate].len());

                    for &rho in &rho_cache[candidate] {
                        let key_id = 50_000usize + digit_index * 10_000 + candidate * 1_000 + rho;
                        let mut xe_seed = [0u8; 32];
                        let mut xa_seed = [0u8; 32];
                        xe_seed[..8].copy_from_slice(&(key_id as u64).to_le_bytes());
                        xa_seed[..8].copy_from_slice(&(key_id as u64).to_le_bytes());
                        xe_seed[8] = 0x5b;
                        xa_seed[8] = 0xb5;

                        let mut key_xe = Source::new(xe_seed);
                        let mut key_xa = Source::new(xa_seed);

                        let key = hmux_rot_packed_key_encrypt_sk(
                            module,
                            host_module,
                            &phase5b_sk_host,
                            &beta_hat_by_candidate[candidate],
                            rho,
                            kk,
                            base2k.into(),
                            mux_dsize,
                            &mut key_xe,
                            &mut key_xa,
                            &mut scratch.borrow(),
                        )
                        .unwrap();

                        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key.key());
                        module.glwe_switching_key_prepare(&mut prepared, key.key(), &mut scratch.borrow());
                        candidate_keys.push(HMuxRotKeyPrepared {
                            key: prepared,
                            gal_el: key.gal_el(),
                        });
                    }

                    key_cache.push(candidate_keys);
                }

                let cached_this_digit: usize = key_cache.iter().map(Vec::len).sum();
                total_cached_keys += cached_this_digit;

                let plans = crate::default::ship::mux::ship_mux_plans(module, key_cache.iter().map(Vec::as_slice));

                let mut next_state = Vec::with_capacity(groups);
                let mut max_rho = 0usize;

                for out_group in 0..groups {
                    let mut source_indices = Vec::with_capacity(base);
                    let mut key_refs = Vec::with_capacity(base);

                    for candidate in 0..base {
                        let runtime_rot = candidate * digit_weight;
                        let route =
                            crate::default::ship::mux::ship_sheared_runtime_route_from_output(out_group, runtime_rot, groups);

                        max_rho = max_rho.max(route.physical_rot);
                        source_indices.push(route.group);

                        let cache_index = rho_cache[candidate]
                            .iter()
                            .position(|&rho| rho == route.physical_rot)
                            .expect("Phase 5B-2 route rho missing from key cache");
                        key_refs.push(&key_cache[candidate][cache_index]);
                    }

                    let source_refs: Vec<&crate::layouts::CKKSCiphertextOwned<BE>> =
                        source_indices.iter().map(|&idx| &state[idx]).collect();

                    let mut out = alloc_ct(&phase5b_params, module, kk);
                    module.ckks_copy(&mut out, source_refs[0], &mut scratch.borrow()).unwrap();

                    let mux_bytes = crate::default::ship::mux::ship_mux_rotate_tmp_bytes(
                        module,
                        source_refs[0],
                        &key_refs[0].key,
                        key_refs.len(),
                    );
                    let mut mux_scratch = ScratchOwned::<BE>::alloc(mux_bytes);

                    crate::default::ship::mux::ship_mux_rotate_multi_source_refs_with_offset(
                        module,
                        &mut out,
                        &source_refs,
                        &key_refs,
                        &plans,
                        -(PACKED_BETA_SCALE_BITS as i64),
                        &mut mux_scratch.borrow(),
                    )
                    .unwrap();

                    next_state.push(out);
                }

                state = next_state;

                let required_bits = if ld < 40 {
                    PHASE5B2_MIN_BITS_FFT
                } else {
                    PHASE5B2_MIN_BITS_EXACT
                };
                assert_packed_matches_ordinary_full_groups!(
                    format!("phase5b2-mux-digit-{digit_index}"),
                    &state,
                    &golden_state,
                    required_bits
                );

                println!(
                    "[sheared phase5b2 cache] backend={} digit={} weight={} base={} unique_keys={} rhos={:?} max_physical_rot={}",
                    std::any::type_name::<BE>(),
                    digit_index,
                    digit_weight,
                    base,
                    cached_this_digit,
                    rho_cache,
                    max_rho,
                );
            }

            let scalar_key_objects = h * digits.iter().map(|&(_, base)| base).sum::<usize>();
            let final_brotmux_required_bits = if ld < 40 {
                PHASE5B2_MIN_BITS_FFT
            } else {
                PHASE5B2_MIN_BITS_EXACT
            };
            assert_packed_matches_ordinary_full_groups!(
                "phase5b2-full-brotmux",
                &state,
                &golden_state,
                final_brotmux_required_bits
            );

            println!(
                "[sheared phase5b2 real-brotmux] backend={} h={} groups={} digits={:?} packed_cached_keys={} scalar_key_objects={} reduction={:.2}x",
                std::any::type_name::<BE>(),
                h,
                groups,
                digits,
                total_cached_keys,
                scalar_key_objects,
                scalar_key_objects as f64 / total_cached_keys as f64,
            );
        }
        // GOLDEN-DIFF: ordinary full-BRotMux leaves.  All later product-tree
        // expectations are repacks/products of these actual Poulpy ciphertexts.
        let golden_leaf_slots: Vec<(Vec<F>, Vec<F>)> = golden_state
            .iter()
            .map(|ct| ckks_decrypt_decode::<BE, F, E>(&phase5b_params, module, &encoder, ct, &phase5b_sk, &mut scratch.borrow()))
            .collect();

        // -----------------------------------------------------------------
        // Phase 5C-1: algebraic sheared product-tree topology.
        //
        // After BRotMux, define
        //   Z_g[p] = f_{p mod G}[p+g].
        //
        // A width-d node A_{g,d} covers d consecutive residue lanes at the
        // same logical slot q=p+g. The recurrence is
        //
        //   A_{g,2d}[p]
        //     = A_{g,d}[p] * Rot_{+d}(A_{g+d,d})[p]
        //
        // because Poulpy Rot_{+d}(x)[p] = x[p-d]. We keep only g multiples of
        // 2d at each level. For G=32 this gives 16+8+4+2+1 = 31
        // multiplications/rotations, depth 5, and five distinct rotations.
        //
        // This phase is deliberately clear/algebraic: it proves the exact
        // topology before adding rotation keys and tensor relinearization.
        // -----------------------------------------------------------------
        {
            const STRIDES: [usize; 5] = [1, 2, 4, 8, 16];
            const PHASE5C1_TOL: f64 = 1.0e-10;

            assert_eq!(groups, 32);
            assert_eq!(groups, 1usize << STRIDES.len());

            let mut level_re: Vec<Option<Vec<F>>> = (0..groups).map(|_| None).collect();
            let mut level_im: Vec<Option<Vec<F>>> = (0..groups).map(|_| None).collect();

            for group in 0..groups {
                let mut re = vec![F::zero(); m];
                let mut im = vec![F::zero(); m];

                for p in 0..m {
                    let lane = p % groups;
                    let q = (p + group) % m;
                    re[p] = golden_leaf_slots[lane].0[q];
                    im[p] = golden_leaf_slots[lane].1[q];
                }

                level_re[group] = Some(re);
                level_im[group] = Some(im);
            }

            let mut total_mults = 0usize;
            let mut worst_err = 0.0f64;
            let mut worst_stride = 0usize;
            let mut worst_group = 0usize;
            let mut worst_slot = 0usize;

            for &stride in &STRIDES {
                let width = 2 * stride;
                let mut next_re: Vec<Option<Vec<F>>> = (0..groups).map(|_| None).collect();
                let mut next_im: Vec<Option<Vec<F>>> = (0..groups).map(|_| None).collect();

                for group in (0..groups).step_by(width) {
                    let right_group = group + stride;
                    assert!(right_group < groups);

                    let left_re = level_re[group].as_ref().expect("Phase 5C-1 missing left subtree");
                    let left_im = level_im[group].as_ref().expect("Phase 5C-1 missing left subtree");
                    let right_re = level_re[right_group].as_ref().expect("Phase 5C-1 missing right subtree");
                    let right_im = level_im[right_group].as_ref().expect("Phase 5C-1 missing right subtree");

                    let mut out_re = vec![F::zero(); m];
                    let mut out_im = vec![F::zero(); m];

                    for p in 0..m {
                        // Poulpy Rot_{+stride}(right)[p] = right[p-stride].
                        let rp = (p + m - stride) % m;

                        let ar = left_re[p];
                        let ai = left_im[p];
                        let br = right_re[rp];
                        let bi = right_im[rp];

                        out_re[p] = ar * br - ai * bi;
                        out_im[p] = ar * bi + ai * br;

                        let q = (p + group) % m;
                        let first_lane = p % groups;
                        let mut wr = F::from_f64(1.0).unwrap();
                        let mut wi = F::zero();

                        for j in 0..width {
                            // Golden leaves come from ordinary Poulpy BRotMux.
                            // This direct clear product checks only the sheared
                            // tree topology; it does not reconstruct SHIP offsets.
                            let factor_lane = (first_lane + groups - j) % groups;
                            let fr = golden_leaf_slots[factor_lane].0[q];
                            let fi = golden_leaf_slots[factor_lane].1[q];

                            let nr = wr * fr - wi * fi;
                            let ni = wr * fi + wi * fr;
                            wr = nr;
                            wi = ni;
                        }

                        let er = (out_re[p].to_f64().unwrap() - wr.to_f64().unwrap()).abs();
                        let ei = (out_im[p].to_f64().unwrap() - wi.to_f64().unwrap()).abs();
                        let local = er.max(ei);

                        if local > worst_err {
                            worst_err = local;
                            worst_stride = stride;
                            worst_group = group;
                            worst_slot = p;
                        }
                    }

                    next_re[group] = Some(out_re);
                    next_im[group] = Some(out_im);
                    total_mults += 1;
                }

                level_re = next_re;
                level_im = next_im;
            }

            assert_eq!(total_mults, groups - 1, "Phase 5C-1 tree must use exactly G-1 products");
            assert!(
                worst_err <= PHASE5C1_TOL,
                "Phase 5C-1 topology mismatch: max_err={:.3e} at stride={} group={} slot={}",
                worst_err,
                worst_stride,
                worst_group,
                worst_slot,
            );

            let root_re = level_re[0].as_ref().expect("Phase 5C-1 missing final root");
            let root_im = level_im[0].as_ref().expect("Phase 5C-1 missing final root");

            let mut root_max_err = 0.0f64;
            let mut root_worst_slot = 0usize;

            for p in 0..m {
                let mut wr = F::from_f64(1.0).unwrap();
                let mut wi = F::zero();

                for lane in 0..groups {
                    let (fr, fi) = (golden_leaf_slots[lane].0[p], golden_leaf_slots[lane].1[p]);

                    let nr = wr * fr - wi * fi;
                    let ni = wr * fi + wi * fr;
                    wr = nr;
                    wi = ni;
                }

                let er = (root_re[p].to_f64().unwrap() - wr.to_f64().unwrap()).abs();
                let ei = (root_im[p].to_f64().unwrap() - wi.to_f64().unwrap()).abs();
                let local = er.max(ei);

                if local > root_max_err {
                    root_max_err = local;
                    root_worst_slot = p;
                }
            }

            assert!(
                root_max_err <= PHASE5C1_TOL,
                "Phase 5C-1 final root mismatch: max_err={:.3e} slot={}",
                root_max_err,
                root_worst_slot,
            );

            println!(
                "[sheared phase5c1 product-tree-layout] backend={} groups={} strides={:?} depth={} multiplications={} rotations={} distinct_rotation_amounts={} stage_max_err={:.3e} root_max_err={:.3e} root_worst_slot={}",
                std::any::type_name::<BE>(),
                groups,
                STRIDES,
                STRIDES.len(),
                total_mults,
                total_mults,
                STRIDES.len(),
                worst_err,
                root_max_err,
                root_worst_slot,
            );
        }
        // -----------------------------------------------------------------
        // Phase 5C-2a: first encrypted sheared product-tree level.
        //
        // Phase 5C-1 proved the index-level recurrence
        //
        //   A_{g,2}[p] = A_{g,1}[p] * A_{g+1,1}[p-1].
        //
        // Here we first pin down Poulpy's public CKKS rotation API convention
        // empirically by generating API k=+1 and the modular inverse-step
        // k=m-1 keys and comparing both outputs against p-1 and p+1 slot
        // oracles. Note: Poulpy's galois_element(-1) is a signed Galois
        // element (-5), not the inverse generator 5^{-1}.
        // that realizes the required p-1 source and exercise the real CKKS path:
        // genuine homomorphic rotation followed by ct×ct multiplication and
        // tensor relinearization for all 16 width-2 tree nodes.
        //
        // These keys are test-local. Production ShipKeySet is intentionally
        // left unchanged until this encrypted gate passes.
        // -----------------------------------------------------------------
        {
            use crate::api::{CKKSConjugateOps as _, CKKSMulOps as _, CKKSRotateOps as _};
            use poulpy_core::layouts::{
                GLWEAutomorphismKeyPreparedFactory as _, GLWETensorKeyPreparedFactory as _, ModuleCoreAlloc as _,
            };
            use poulpy_core::{GLWEAutomorphismKeyEncryptSk as _, GLWETensorKeyEncryptSk as _, TransferInto as _};
            use poulpy_hal::layouts::GaloisElement as _;

            const PHASE5C2A_MIN_BITS_FFT: f64 = 8.0;
            const PHASE5C2A_MIN_BITS_EXACT: f64 = 12.0;

            assert_eq!(groups, 32);
            assert_eq!(state.len(), groups);

            let golden_leaf_state = |group: usize, p: usize| -> (F, F) {
                let lane = p % groups;
                let q = (p + group) % m;
                (golden_leaf_slots[lane].0[q], golden_leaf_slots[lane].1[q])
            };

            // Build test-local evaluation keys at the actual raised width kk.
            let mut tree_key_params = phase5b_params;
            tree_key_params.k = kk;
            let mut tree_scratch = alloc_scratch(&tree_key_params, module);

            let mut tree_sk = module.glwe_secret_alloc_from_infos(&phase5b_sk_host);
            phase5b_sk_host.transfer_into(&mut tree_sk);

            let mut key_xe = Source::new([0x51u8; 32]);
            let mut key_xa = Source::new([0xa1u8; 32]);

            let atk_infos = tree_key_params.atk_layout();

            // Final real-SHIP recombination uses root + Conj(root).
            // Production ShipKeySet also generates this automorphism key at -1.
            let mut ship_conj_key = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
            module.glwe_automorphism_key_encrypt_sk(
                &mut ship_conj_key,
                -1,
                &tree_sk,
                &atk_infos,
                &mut key_xe,
                &mut key_xa,
                &mut tree_scratch.borrow(),
            );
            let mut ship_conj_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&ship_conj_key);
            module.glwe_automorphism_key_prepare(&mut ship_conj_prepared, &ship_conj_key, &mut tree_scratch.borrow());

            let mut rot_pos1_key = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
            let rot_pos1_gal = module.galois_element(1);
            module.glwe_automorphism_key_encrypt_sk(
                &mut rot_pos1_key,
                rot_pos1_gal,
                &tree_sk,
                &atk_infos,
                &mut key_xe,
                &mut key_xa,
                &mut tree_scratch.borrow(),
            );
            let mut rot_pos1_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&rot_pos1_key);
            module.glwe_automorphism_key_prepare(&mut rot_pos1_prepared, &rot_pos1_key, &mut tree_scratch.borrow());

            let mut rot_back1_key = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
            let rot_back1_api = (m - 1) as i64;
            let rot_back1_gal = module.galois_element(rot_back1_api);
            module.glwe_automorphism_key_encrypt_sk(
                &mut rot_back1_key,
                rot_back1_gal,
                &tree_sk,
                &atk_infos,
                &mut key_xe,
                &mut key_xa,
                &mut tree_scratch.borrow(),
            );
            let mut rot_back1_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&rot_back1_key);
            module.glwe_automorphism_key_prepare(&mut rot_back1_prepared, &rot_back1_key, &mut tree_scratch.borrow());

            let tsk_infos = tree_key_params.tsk_layout();
            let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
            module.glwe_tensor_key_encrypt_sk(
                &mut tensor_key,
                &tree_sk,
                &tsk_infos,
                &mut key_xe,
                &mut key_xa,
                &mut tree_scratch.borrow(),
            );

            let mut tensor_prepared = module.alloc_tensor_key_prepared_from_infos(&tensor_key);
            module.prepare_tensor_key(&mut tensor_prepared, &tensor_key, &mut tree_scratch.borrow());

            // GOLDEN-DIFF product-tree reference.  At width d, ordinary node r
            // is the canonical-slot product of d cyclic leaves ending at lane r:
            //     W[r,d] = product_{t=0..d-1} leaf[(r-t) mod G].
            // It is generated only with ordinary Poulpy ckks_mul_into and the
            // production SHIP tensor key; no sheared rotation formula is used.
            macro_rules! golden_cyclic_next {
                ($current:expr, $width:expr) => {{
                    let current_width = $width;
                    let out_width = 2 * current_width;
                    let mut next = Vec::with_capacity(groups);
                    let mut golden_self_worst = 0.0f64;
                    let mut golden_self_worst_lane = 0usize;
                    let mut golden_self_worst_slot = 0usize;
                    let mut golden_self_max_abs = 0.0f64;

                    for lane in 0..groups {
                        let right_lane = (lane + groups - (current_width % groups)) % groups;
                        let left = &$current[lane];
                        let right = &$current[right_lane];

                        // Differential probe against the ordinary operands
                        // themselves.  This is deliberately independent of the
                        // packed/sheared path: it asks whether chained Poulpy
                        // ckks_mul_into remains semantically correct at this
                        // width on the current backend.
                        let (left_re, left_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            left,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );
                        let (right_re, right_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            right,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let budget = left.log_budget().min(right.log_budget());
                        let consumed = left.log_delta().max(right.log_delta());
                        assert!(budget >= consumed, "golden cyclic product tree exhausted budget");
                        let k_dst = budget - consumed + left.log_delta().min(right.log_delta());
                        let mut dst = alloc_ct(&phase5b_params, module, k_dst);
                        module
                            .ckks_mul_into(
                                &mut dst,
                                left,
                                right,
                                golden_keys.tensor_key(),
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();

                        let (dst_re, dst_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &dst,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        for p in 0..m {
                            let ar = left_re[p].to_f64().unwrap();
                            let ai = left_im[p].to_f64().unwrap();
                            let br = right_re[p].to_f64().unwrap();
                            let bi = right_im[p].to_f64().unwrap();
                            let want_re = ar * br - ai * bi;
                            let want_im = ar * bi + ai * br;
                            let got_re = dst_re[p].to_f64().unwrap();
                            let got_im = dst_im[p].to_f64().unwrap();
                            let local = (got_re - want_re).abs().max((got_im - want_im).abs());
                            golden_self_max_abs = golden_self_max_abs.max(got_re.abs()).max(got_im.abs());
                            if local > golden_self_worst {
                                golden_self_worst = local;
                                golden_self_worst_lane = lane;
                                golden_self_worst_slot = p;
                            }
                        }

                        next.push(dst);
                    }

                    let golden_self_bits = if golden_self_worst == 0.0 {
                        f64::INFINITY
                    } else {
                        -golden_self_worst.log2()
                    };
                    println!(
                        "[golden-self tree-width-{}] backend={} max_err={:.3e} ({:.2} bits) max_abs={:.3e} worst_lane={} worst_slot={}",
                        out_width,
                        std::any::type_name::<BE>(),
                        golden_self_worst,
                        golden_self_bits,
                        golden_self_max_abs,
                        golden_self_worst_lane,
                        golden_self_worst_slot,
                    );

                    next
                }};
            }

            macro_rules! assert_packed_tree_matches_golden {
                ($tag:expr, $packed:expr, $width:expr, $golden:expr, $required_bits:expr) => {{
                    let tag = $tag;
                    let width_now = $width;
                    assert_eq!(($packed).len(), groups / width_now);
                    assert_eq!(($golden).len(), groups);

                    let golden_slots: Vec<(Vec<F>, Vec<F>)> = ($golden)
                        .iter()
                        .map(|ct| {
                            ckks_decrypt_decode::<BE, F, E>(
                                &phase5b_params,
                                module,
                                &encoder,
                                ct,
                                &phase5b_sk,
                                &mut tree_scratch.borrow(),
                            )
                        })
                        .collect();

                    let mut worst = 0.0f64;
                    let mut worst_group = 0usize;
                    let mut worst_slot = 0usize;
                    let mut sum_sq = 0.0f64;
                    let mut count = 0usize;

                    for (node_index, node) in ($packed).iter().enumerate() {
                        let group = node_index * width_now;
                        let (have_re, have_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            node,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );
                        for p in 0..m {
                            let lane = p % groups;
                            let q = (p + group) % m;
                            let want_re = golden_slots[lane].0[q];
                            let want_im = golden_slots[lane].1[q];
                            let er = (have_re[p].to_f64().unwrap() - want_re.to_f64().unwrap()).abs();
                            let ei = (have_im[p].to_f64().unwrap() - want_im.to_f64().unwrap()).abs();
                            let local = er.max(ei);
                            if local > worst {
                                worst = local;
                                worst_group = group;
                                worst_slot = p;
                            }
                            sum_sq += er * er + ei * ei;
                            count += 2;
                        }
                    }

                    let bits = if worst == 0.0 { f64::INFINITY } else { -worst.log2() };
                    let rms = (sum_sq / count as f64).sqrt();
                    println!(
                        "[golden-diff {}] backend={} width={} max_err={:.3e} ({:.2} bits) rms={:.3e} worst_group={} worst_slot={}",
                        tag,
                        std::any::type_name::<BE>(),
                        width_now,
                        worst,
                        bits,
                        rms,
                        worst_group,
                        worst_slot,
                    );
                    assert!(
                        bits >= $required_bits,
                        "golden-diff {} failed at width {}: {:.2} bits < {:.2} bits (max_err={:.3e}, group={}, slot={})",
                        tag,
                        width_now,
                        bits,
                        $required_bits,
                        worst,
                        worst_group,
                        worst_slot,
                    );
                }};
            }

            let mut golden_cyclic = golden_cyclic_next!(golden_state, 1usize);

            // Independent convention probe on group 1. The product tree
            // requires source slot p-1, but we deliberately test both API signs
            // before assuming which public CKKS parameter realizes it. In this
            // implementation the backward one-slot candidate is k=m-1, not k=-1.
            let mut rot_pos1_probe = alloc_ct(&phase5b_params, module, state[1].k().as_usize());
            module
                .ckks_rotate_into(
                    &mut rot_pos1_probe,
                    &state[1],
                    1,
                    &rot_pos1_prepared,
                    &mut tree_scratch.borrow(),
                )
                .unwrap();

            let mut rot_back1_probe = alloc_ct(&phase5b_params, module, state[1].k().as_usize());
            module
                .ckks_rotate_into(
                    &mut rot_back1_probe,
                    &state[1],
                    rot_back1_api,
                    &rot_back1_prepared,
                    &mut tree_scratch.borrow(),
                )
                .unwrap();

            for probe in [&rot_pos1_probe, &rot_back1_probe] {
                assert_eq!(
                    probe.log_delta(),
                    state[1].log_delta(),
                    "Phase 5C-2a rotation changed log_delta"
                );
                assert_eq!(
                    probe.log_budget(),
                    state[1].log_budget(),
                    "Phase 5C-2a equal-width rotation changed log_budget"
                );
            }

            let (pos_re, pos_im) = ckks_decrypt_decode::<BE, F, E>(
                &phase5b_params,
                module,
                &encoder,
                &rot_pos1_probe,
                &phase5b_sk,
                &mut tree_scratch.borrow(),
            );
            let (back_re, back_im) = ckks_decrypt_decode::<BE, F, E>(
                &phase5b_params,
                module,
                &encoder,
                &rot_back1_probe,
                &phase5b_sk,
                &mut tree_scratch.borrow(),
            );

            let mut pos_to_minus_err = 0.0f64;
            let mut pos_to_plus_err = 0.0f64;
            let mut back_to_minus_err = 0.0f64;
            let mut back_to_plus_err = 0.0f64;
            let mut pos_to_minus_slot = 0usize;
            let mut back_to_minus_slot = 0usize;

            for p in 0..m {
                let src_minus = (p + m - 1) % m;
                let src_plus = (p + 1) % m;
                let (want_minus_re, want_minus_im) = golden_leaf_state(1, src_minus);
                let (want_plus_re, want_plus_im) = golden_leaf_state(1, src_plus);

                let pos_minus = (pos_re[p].to_f64().unwrap() - want_minus_re.to_f64().unwrap())
                    .abs()
                    .max((pos_im[p].to_f64().unwrap() - want_minus_im.to_f64().unwrap()).abs());
                let pos_plus = (pos_re[p].to_f64().unwrap() - want_plus_re.to_f64().unwrap())
                    .abs()
                    .max((pos_im[p].to_f64().unwrap() - want_plus_im.to_f64().unwrap()).abs());
                let neg_minus = (back_re[p].to_f64().unwrap() - want_minus_re.to_f64().unwrap())
                    .abs()
                    .max((back_im[p].to_f64().unwrap() - want_minus_im.to_f64().unwrap()).abs());
                let neg_plus = (back_re[p].to_f64().unwrap() - want_plus_re.to_f64().unwrap())
                    .abs()
                    .max((back_im[p].to_f64().unwrap() - want_plus_im.to_f64().unwrap()).abs());

                if pos_minus > pos_to_minus_err {
                    pos_to_minus_err = pos_minus;
                    pos_to_minus_slot = p;
                }
                pos_to_plus_err = pos_to_plus_err.max(pos_plus);

                if neg_minus > back_to_minus_err {
                    back_to_minus_err = neg_minus;
                    back_to_minus_slot = p;
                }
                back_to_plus_err = back_to_plus_err.max(neg_plus);
            }

            let min_bits = if ld < 40 {
                PHASE5C2A_MIN_BITS_FFT
            } else {
                PHASE5C2A_MIN_BITS_EXACT
            };

            let (tree_rotation, tree_rotation_key, rotation_max_err, rotation_worst_slot) =
                if back_to_minus_err <= pos_to_minus_err {
                    (rot_back1_api, &rot_back1_prepared, back_to_minus_err, back_to_minus_slot)
                } else {
                    (1i64, &rot_pos1_prepared, pos_to_minus_err, pos_to_minus_slot)
                };

            let rotation_bits = if rotation_max_err == 0.0 {
                f64::INFINITY
            } else {
                -rotation_max_err.log2()
            };

            println!(
                "[sheared phase5c2a rotation-convention] backend={} api+1->p-1={:.3e} api+1->p+1={:.3e} api(m-1={})->p-1={:.3e} api(m-1)->p+1={:.3e} selected_api_k={} selected_bits={:.1}",
                std::any::type_name::<BE>(),
                pos_to_minus_err,
                pos_to_plus_err,
                rot_back1_api,
                back_to_minus_err,
                back_to_plus_err,
                tree_rotation,
                rotation_bits,
            );

            assert!(
                rotation_bits >= min_bits,
                "ship_sheared_phase5c2a_rotation: neither +1 nor modular inverse-step realizes p-1 with enough precision; selected k={} precision {:.1} bits < {:.1} bits (max_err={:.3e}, slot={})",
                tree_rotation,
                rotation_bits,
                min_bits,
                rotation_max_err,
                rotation_worst_slot,
            );

            let mut level1 = Vec::with_capacity(groups / 2);
            let mut worst_mul_err = 0.0f64;
            let mut worst_mul_bits = f64::INFINITY;
            let mut worst_group = 0usize;
            let mut worst_slot = 0usize;
            let mut total_sq_err = 0.0f64;
            let mut total_components = 0usize;
            let mut output_log_budget = None;

            for group in (0..groups).step_by(2) {
                let right_group = group + 1;

                let mut rotated = alloc_ct(&phase5b_params, module, state[right_group].k().as_usize());
                {
                    module
                        .ckks_rotate_into(
                            &mut rotated,
                            &state[right_group],
                            tree_rotation,
                            tree_rotation_key,
                            &mut tree_scratch.borrow(),
                        )
                        .unwrap();
                }

                let budget = state[group].log_budget().min(rotated.log_budget());
                let consumed = state[group].log_delta().max(rotated.log_delta());
                assert!(
                    budget >= consumed,
                    "Phase 5C-2a first product level exhausts budget: group={group} budget={budget} consumed={consumed}"
                );
                let k_dst = budget - consumed + state[group].log_delta().min(rotated.log_delta());

                let mut out = alloc_ct(&phase5b_params, module, k_dst);

                module
                    .ckks_mul_into(
                        &mut out,
                        &state[group],
                        &rotated,
                        &tensor_prepared,
                        &mut tree_scratch.borrow(),
                    )
                    .unwrap();

                if let Some(previous) = output_log_budget {
                    assert_eq!(previous, out.log_budget(), "Phase 5C-2a pair outputs diverged in budget");
                } else {
                    output_log_budget = Some(out.log_budget());
                }

                let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(
                    &phase5b_params,
                    module,
                    &encoder,
                    &out,
                    &phase5b_sk,
                    &mut tree_scratch.borrow(),
                );

                let mut pair_max_err = 0.0f64;
                let mut pair_worst_slot = 0usize;

                for p in 0..m {
                    let src_p = (p + m - 1) % m;
                    let (ar, ai) = golden_leaf_state(group, p);
                    let (br, bi) = golden_leaf_state(right_group, src_p);
                    let wr = ar * br - ai * bi;
                    let wi = ar * bi + ai * br;

                    let er = (got_re[p].to_f64().unwrap() - wr.to_f64().unwrap()).abs();
                    let ei = (got_im[p].to_f64().unwrap() - wi.to_f64().unwrap()).abs();
                    let local = er.max(ei);

                    if local > pair_max_err {
                        pair_max_err = local;
                        pair_worst_slot = p;
                    }
                    total_sq_err += er * er + ei * ei;
                    total_components += 2;
                }

                let pair_bits = if pair_max_err == 0.0 {
                    f64::INFINITY
                } else {
                    -pair_max_err.log2()
                };

                if pair_bits < min_bits {
                    println!(
                        "[diagnostic-only phase5c2a formula-oracle] group={} precision={:.2} bits max_err={:.3e} slot={}",
                        group, pair_bits, pair_max_err, pair_worst_slot,
                    );
                }

                if pair_max_err > worst_mul_err {
                    worst_mul_err = pair_max_err;
                    worst_mul_bits = pair_bits;
                    worst_group = group;
                    worst_slot = pair_worst_slot;
                }

                level1.push(out);
            }

            assert_eq!(level1.len(), 16);
            assert_packed_tree_matches_golden!("phase5c2a-width2", &level1, 2usize, &golden_cyclic, min_bits);

            let global_rms = (total_sq_err / total_components as f64).sqrt();
            let global_rms_bits = if global_rms == 0.0 {
                f64::INFINITY
            } else {
                -global_rms.log2()
            };

            println!(
                "[sheared phase5c2a encrypted-tree-level1] backend={} pairs={} api_rotation={} rotation_max_err={:.3e} ({:.1} bits) rotation_worst_slot={} mul_worst_group={} mul_worst_slot={} mul_worst_max_err={:.3e} ({:.1} bits) global_rms={:.3e} ({:.1} bits) input_log_budget={} output_log_budget={}",
                std::any::type_name::<BE>(),
                level1.len(),
                tree_rotation,
                rotation_max_err,
                rotation_bits,
                rotation_worst_slot,
                worst_group,
                worst_slot,
                worst_mul_err,
                worst_mul_bits,
                global_rms,
                global_rms_bits,
                state[0].log_budget(),
                output_log_budget.unwrap(),
            );
            // -------------------------------------------------------------
            // Phase 5C-2b: complete encrypted sheared product tree.
            //
            // Phase 5C-2a produced the width-2 nodes. Continue with widths
            // 2 -> 4 -> 8 -> 16 -> 32 using the proven index recurrence
            //
            //   A_{g,2d}[p] = A_{g,d}[p] * A_{g+d,d}[p-d].
            //
            // Poulpy's public CKKS API maps the required p-d source to
            // k = m-d (mod m), not to signed k=-d.
            //
            // Every stage uses one automorphism key shared by all nodes at
            // that stride and the same tensor/relinearization key.
            // -------------------------------------------------------------
            assert_eq!(
                tree_rotation, rot_back1_api,
                "Phase 5C-2b expects the modular inverse-cycle rotation chosen in Phase 5C-2a"
            );

            let initial_tree_budget = state[0].log_budget();
            let tree_log_delta = state[0].log_delta();
            let mut current = level1;
            let mut width = 2usize;
            let mut tree_rotations = groups / 2;
            let mut tree_multiplications = groups / 2;
            let mut distinct_tree_rotation_keys = 1usize;

            // Keep this gate deliberately loose: Phase 5C-2b is intended to
            // expose the full depth/noise trajectory. A topology/sign error
            // gives O(1) error, while valid CKKS noise should remain far below.
            const PHASE5C2B_MAX_ABS_ERR: f64 = 0.25;

            while width < groups {
                assert_eq!(
                    current.len(),
                    groups / width,
                    "Phase 5C-2b node count does not match current width"
                );

                let shift = width;
                let api_rotation = ((m - (shift % m)) % m) as i64;
                assert_ne!(api_rotation, 0, "Phase 5C-2b unexpectedly requested the identity rotation");

                let mut stage_rot_key = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
                let stage_gal = module.galois_element(api_rotation);
                module.glwe_automorphism_key_encrypt_sk(
                    &mut stage_rot_key,
                    stage_gal,
                    &tree_sk,
                    &atk_infos,
                    &mut key_xe,
                    &mut key_xa,
                    &mut tree_scratch.borrow(),
                );
                let mut stage_rot_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&stage_rot_key);
                module.glwe_automorphism_key_prepare(&mut stage_rot_prepared, &stage_rot_key, &mut tree_scratch.borrow());

                distinct_tree_rotation_keys += 1;

                let input_stage_budget = current[0].log_budget();
                let input_stage_delta = current[0].log_delta();
                for node in &current {
                    assert_eq!(
                        node.log_budget(),
                        input_stage_budget,
                        "Phase 5C-2b stage inputs diverged in budget"
                    );
                    assert_eq!(
                        node.log_delta(),
                        input_stage_delta,
                        "Phase 5C-2b stage inputs diverged in scale"
                    );
                }

                let out_width = 2 * width;
                let mut next = Vec::with_capacity(current.len() / 2);
                let mut stage_worst_err = 0.0f64;
                let mut stage_worst_group = 0usize;
                let mut stage_worst_slot = 0usize;
                let mut stage_sq_err = 0.0f64;
                let mut stage_components = 0usize;
                let mut stage_output_budget = None;

                for pair in 0..(current.len() / 2) {
                    let left_index = 2 * pair;
                    let right_index = left_index + 1;
                    let group = pair * out_width;

                    let left = &current[left_index];
                    let right = &current[right_index];

                    let mut rotated = alloc_ct(&phase5b_params, module, right.k().as_usize());
                    module
                        .ckks_rotate_into(
                            &mut rotated,
                            right,
                            api_rotation,
                            &stage_rot_prepared,
                            &mut tree_scratch.borrow(),
                        )
                        .unwrap();

                    assert_eq!(
                        rotated.log_delta(),
                        right.log_delta(),
                        "Phase 5C-2b rotation changed log_delta at stride {shift}"
                    );
                    assert_eq!(
                        rotated.log_budget(),
                        right.log_budget(),
                        "Phase 5C-2b rotation changed log_budget at stride {shift}"
                    );

                    let budget = left.log_budget().min(rotated.log_budget());
                    let consumed = left.log_delta().max(rotated.log_delta());
                    assert!(
                        budget >= consumed,
                        "Phase 5C-2b exhausts budget: width={width} group={group} budget={budget} consumed={consumed}"
                    );
                    let k_dst = budget - consumed + left.log_delta().min(rotated.log_delta());

                    let mut out = alloc_ct(&phase5b_params, module, k_dst);
                    module
                        .ckks_mul_into(&mut out, left, &rotated, &tensor_prepared, &mut tree_scratch.borrow())
                        .unwrap();

                    if let Some(previous) = stage_output_budget {
                        assert_eq!(previous, out.log_budget(), "Phase 5C-2b stage outputs diverged in budget");
                    } else {
                        stage_output_budget = Some(out.log_budget());
                    }

                    let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(
                        &phase5b_params,
                        module,
                        &encoder,
                        &out,
                        &phase5b_sk,
                        &mut tree_scratch.borrow(),
                    );

                    let mut pair_max_err = 0.0f64;
                    let mut pair_worst_slot = 0usize;

                    for p in 0..m {
                        let (mut wr, mut wi) = golden_leaf_state(group, p);
                        for j in 1..out_width {
                            let src_p = (p + m - (j % m)) % m;
                            let (br, bi) = golden_leaf_state(group + j, src_p);
                            let nr = wr * br - wi * bi;
                            let ni = wr * bi + wi * br;
                            wr = nr;
                            wi = ni;
                        }

                        let er = (got_re[p].to_f64().unwrap() - wr.to_f64().unwrap()).abs();
                        let ei = (got_im[p].to_f64().unwrap() - wi.to_f64().unwrap()).abs();
                        let local = er.max(ei);

                        if local > pair_max_err {
                            pair_max_err = local;
                            pair_worst_slot = p;
                        }

                        stage_sq_err += er * er + ei * ei;
                        stage_components += 2;
                    }

                    if pair_max_err > stage_worst_err {
                        stage_worst_err = pair_max_err;
                        stage_worst_group = group;
                        stage_worst_slot = pair_worst_slot;
                    }

                    if out_width == 8 && pair == 0 && std::any::type_name::<BE>().to_ascii_lowercase().contains("fft64") {
                        let (left_re, left_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            left,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );
                        let (right_re, right_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            right,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );
                        let (rot_re, rot_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &rotated,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut left_oracle_err = 0.0f64;
                        let mut right_oracle_err = 0.0f64;
                        let mut rotation_vs_decoded_err = 0.0f64;
                        let mut mul_vs_decoded_err = 0.0f64;
                        let mut max_abs_left = 0.0f64;
                        let mut max_abs_right = 0.0f64;
                        let mut max_abs_rot = 0.0f64;
                        let mut max_abs_decoded_product = 0.0f64;
                        let mut max_abs_out = 0.0f64;

                        for p in 0..m {
                            let subtree_oracle = |base_group: usize, slot: usize| -> (f64, f64) {
                                let (wr0, wi0) = golden_leaf_state(base_group, slot);
                                let mut wr = wr0.to_f64().unwrap();
                                let mut wi = wi0.to_f64().unwrap();
                                for j in 1..width {
                                    let src_p = (slot + m - (j % m)) % m;
                                    let (br0, bi0) = golden_leaf_state(base_group + j, src_p);
                                    let br = br0.to_f64().unwrap();
                                    let bi = bi0.to_f64().unwrap();
                                    let nr = wr * br - wi * bi;
                                    let ni = wr * bi + wi * br;
                                    wr = nr;
                                    wi = ni;
                                }
                                (wr, wi)
                            };

                            let lr = left_re[p].to_f64().unwrap();
                            let li = left_im[p].to_f64().unwrap();
                            let rr = right_re[p].to_f64().unwrap();
                            let ri = right_im[p].to_f64().unwrap();
                            let zr = rot_re[p].to_f64().unwrap();
                            let zi = rot_im[p].to_f64().unwrap();
                            let gr = got_re[p].to_f64().unwrap();
                            let gi = got_im[p].to_f64().unwrap();

                            let (wl_r, wl_i) = subtree_oracle(group, p);
                            let (wr_r, wr_i) = subtree_oracle(group + width, p);
                            left_oracle_err = left_oracle_err.max((lr - wl_r).abs()).max((li - wl_i).abs());
                            right_oracle_err = right_oracle_err.max((rr - wr_r).abs()).max((ri - wr_i).abs());

                            let src = (p + m - (shift % m)) % m;
                            let rr_shift = right_re[src].to_f64().unwrap();
                            let ri_shift = right_im[src].to_f64().unwrap();
                            rotation_vs_decoded_err =
                                rotation_vs_decoded_err.max((zr - rr_shift).abs()).max((zi - ri_shift).abs());

                            let pr = lr * zr - li * zi;
                            let pi = lr * zi + li * zr;
                            mul_vs_decoded_err = mul_vs_decoded_err.max((gr - pr).abs()).max((gi - pi).abs());

                            max_abs_left = max_abs_left.max(lr.abs()).max(li.abs());
                            max_abs_right = max_abs_right.max(rr.abs()).max(ri.abs());
                            max_abs_rot = max_abs_rot.max(zr.abs()).max(zi.abs());
                            max_abs_decoded_product = max_abs_decoded_product.max(pr.abs()).max(pi.abs());
                            max_abs_out = max_abs_out.max(gr.abs()).max(gi.abs());
                        }

                        // Control: same-width ct×ct multiplication without the
                        // automorphism output. If this also fails, the problem is
                        // in the FFT multiplication/relinearization path at this
                        // stage rather than in the rotation ciphertext.
                        let mut control = alloc_ct(&phase5b_params, module, k_dst);
                        module
                            .ckks_mul_into(&mut control, left, right, &tensor_prepared, &mut tree_scratch.borrow())
                            .unwrap();
                        let (control_re, control_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &control,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut control_mul_vs_decoded_err = 0.0f64;
                        let mut max_abs_control = 0.0f64;
                        for p in 0..m {
                            let lr = left_re[p].to_f64().unwrap();
                            let li = left_im[p].to_f64().unwrap();
                            let rr = right_re[p].to_f64().unwrap();
                            let ri = right_im[p].to_f64().unwrap();
                            let pr = lr * rr - li * ri;
                            let pi = lr * ri + li * rr;
                            let cr = control_re[p].to_f64().unwrap();
                            let ci = control_im[p].to_f64().unwrap();
                            control_mul_vs_decoded_err = control_mul_vs_decoded_err.max((cr - pr).abs()).max((ci - pi).abs());
                            max_abs_control = max_abs_control.max(cr.abs()).max(ci.abs());
                        }

                        // ---- diagnostic: least-squares complex scale between the
                        // expected product and the decoded control output. If the
                        // decoded output is the expected product times a fixed
                        // complex factor, this reports it directly.
                        {
                            let (mut num_re, mut num_im, mut den) = (0.0f64, 0.0f64, 0.0f64);
                            let mut got_pow = 0.0f64;
                            for p in 0..m {
                                let lr = left_re[p].to_f64().unwrap();
                                let li = left_im[p].to_f64().unwrap();
                                let rr = right_re[p].to_f64().unwrap();
                                let ri = right_im[p].to_f64().unwrap();
                                let pr = lr * rr - li * ri;
                                let pi = lr * ri + li * rr;
                                let cr = control_re[p].to_f64().unwrap();
                                let ci = control_im[p].to_f64().unwrap();
                                num_re += pr * cr + pi * ci;
                                num_im += pr * ci - pi * cr;
                                den += pr * pr + pi * pi;
                                got_pow += cr * cr + ci * ci;
                            }
                            let (s_re, s_im) = if den > 0.0 { (num_re / den, num_im / den) } else { (0.0, 0.0) };
                            let resid = (got_pow - 2.0 * (s_re * num_re - s_im * num_im) + (s_re * s_re + s_im * s_im) * den)
                                .max(0.0)
                                .sqrt();
                            println!(
                                "[diag ctrl-fit] backend={} den={:.3e} got_pow={:.3e} scale=({:.6e},{:.6e}) scale_log2={:.3} resid={:.3e} resid_log2={:.3}",
                                std::any::type_name::<BE>(),
                                den,
                                got_pow,
                                s_re,
                                s_im,
                                (s_re * s_re + s_im * s_im).sqrt().log2(),
                                resid,
                                resid.log2(),
                            );
                            for p in [0usize, 1, 2, 113, 127] {
                                let lr = left_re[p].to_f64().unwrap();
                                let li = left_im[p].to_f64().unwrap();
                                let rr = right_re[p].to_f64().unwrap();
                                let ri = right_im[p].to_f64().unwrap();
                                let pr = lr * rr - li * ri;
                                let pi = lr * ri + li * rr;
                                let cr = control_re[p].to_f64().unwrap();
                                let ci = control_im[p].to_f64().unwrap();
                                println!(
                                    "[diag ctrl-slot] p={} exp=({:.6e},{:.6e}) got=({:.6e},{:.6e}) got_over_exp_mag={:.3e}",
                                    p,
                                    pr,
                                    pi,
                                    cr,
                                    ci,
                                    if pr == 0.0 && pi == 0.0 {
                                        f64::NAN
                                    } else {
                                        (cr * cr + ci * ci).sqrt() / (pr * pr + pi * pi).sqrt()
                                    }
                                );
                            }
                        }

                        // Additional isolate: repeat both products into a destination
                        // physically allocated at the input width instead of the
                        // natural output width. Metadata should still settle to the
                        // natural product precision; only the backing capacity differs.
                        //
                        // If these wide-destination products are correct while the
                        // normal k_dst products explode, the fault is specifically in
                        // the narrower-output tensor/relinearization path.
                        let wide_dst_k = left.k().as_usize();

                        let mut wide_rot_mul = alloc_ct(&phase5b_params, module, wide_dst_k);
                        module
                            .ckks_mul_into(
                                &mut wide_rot_mul,
                                left,
                                &rotated,
                                &tensor_prepared,
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();
                        let (wide_rot_re, wide_rot_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &wide_rot_mul,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut wide_control = alloc_ct(&phase5b_params, module, wide_dst_k);
                        module
                            .ckks_mul_into(&mut wide_control, left, right, &tensor_prepared, &mut tree_scratch.borrow())
                            .unwrap();
                        let (wide_ctl_re, wide_ctl_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &wide_control,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut wide_rot_vs_decoded_err = 0.0f64;
                        let mut wide_ctl_vs_decoded_err = 0.0f64;
                        let mut max_abs_wide_rot = 0.0f64;
                        let mut max_abs_wide_ctl = 0.0f64;

                        for p in 0..m {
                            let lr = left_re[p].to_f64().unwrap();
                            let li = left_im[p].to_f64().unwrap();
                            let rr = right_re[p].to_f64().unwrap();
                            let ri = right_im[p].to_f64().unwrap();
                            let zr = rot_re[p].to_f64().unwrap();
                            let zi = rot_im[p].to_f64().unwrap();

                            let rot_pr = lr * zr - li * zi;
                            let rot_pi = lr * zi + li * zr;
                            let ctl_pr = lr * rr - li * ri;
                            let ctl_pi = lr * ri + li * rr;

                            let wr = wide_rot_re[p].to_f64().unwrap();
                            let wi = wide_rot_im[p].to_f64().unwrap();
                            let cr = wide_ctl_re[p].to_f64().unwrap();
                            let ci = wide_ctl_im[p].to_f64().unwrap();

                            wide_rot_vs_decoded_err = wide_rot_vs_decoded_err.max((wr - rot_pr).abs()).max((wi - rot_pi).abs());
                            wide_ctl_vs_decoded_err = wide_ctl_vs_decoded_err.max((cr - ctl_pr).abs()).max((ci - ctl_pi).abs());

                            max_abs_wide_rot = max_abs_wide_rot.max(wr.abs()).max(wi.abs());
                            max_abs_wide_ctl = max_abs_wide_ctl.max(cr.abs()).max(ci.abs());
                        }

                        println!(
                            "[sheared phase5c2b width8-wide-dst] backend={} normal_dst_k={} wide_alloc_k={} wide_rot_result_k={} wide_rot_budget={} wide_ctl_result_k={} wide_ctl_budget={} wide_rot_vs_decoded_err={:.3e} wide_ctl_vs_decoded_err={:.3e} max_abs_wide_rot={:.3e} max_abs_wide_ctl={:.3e}",
                            std::any::type_name::<BE>(),
                            k_dst,
                            wide_dst_k,
                            wide_rot_mul.k().as_usize(),
                            wide_rot_mul.log_budget(),
                            wide_control.k().as_usize(),
                            wide_control.log_budget(),
                            wide_rot_vs_decoded_err,
                            wide_ctl_vs_decoded_err,
                            max_abs_wide_rot,
                            max_abs_wide_ctl,
                        );

                        // Fresh-ciphertext control: decrypt the two width-4 inputs
                        // (and the already-correct rotated right input), re-encrypt
                        // those slot values at the same logical input width k=176,
                        // then repeat the width-8 products with the same tensor key.
                        //
                        // This distinguishes:
                        //   A) a generic FFT64 multiplication failure at this k, from
                        //   B) a bad internal representation carried by the prior
                        //      homomorphic product outputs.
                        let fresh_k = left.k().as_usize();

                        // ckks_encrypt() builds/encodes its plaintext using
                        // params.k.  Therefore params.k must match the requested
                        // ciphertext width; otherwise the helper creates a
                        // full-width plaintext (here k=266) and cannot align it
                        // with the fresh k=176 zero encryption.
                        let mut fresh_params = phase5b_params;
                        fresh_params.k = fresh_k;
                        fresh_params.prec_log_budget = fresh_k
                            .checked_sub(fresh_params.prec_meta.log_delta)
                            .expect("fresh ciphertext width must cover log_delta");

                        let fresh_left = ckks_encrypt(
                            &fresh_params,
                            module,
                            host_module,
                            &encoder,
                            &phase5b_sk,
                            fresh_k,
                            &left_re,
                            &left_im,
                            &mut tree_scratch.borrow(),
                        );
                        let fresh_right = ckks_encrypt(
                            &fresh_params,
                            module,
                            host_module,
                            &encoder,
                            &phase5b_sk,
                            fresh_k,
                            &right_re,
                            &right_im,
                            &mut tree_scratch.borrow(),
                        );
                        let fresh_rot = ckks_encrypt(
                            &fresh_params,
                            module,
                            host_module,
                            &encoder,
                            &phase5b_sk,
                            fresh_k,
                            &rot_re,
                            &rot_im,
                            &mut tree_scratch.borrow(),
                        );

                        assert_eq!(fresh_left.k().as_usize(), fresh_k);
                        assert_eq!(fresh_right.k().as_usize(), fresh_k);
                        assert_eq!(fresh_rot.k().as_usize(), fresh_k);
                        assert_eq!(fresh_left.log_delta(), left.log_delta());
                        assert_eq!(fresh_left.log_budget(), left.log_budget());

                        let fresh_budget = fresh_left.log_budget().min(fresh_right.log_budget());
                        let fresh_consumed = fresh_left.log_delta().max(fresh_right.log_delta());
                        assert!(fresh_budget >= fresh_consumed);
                        let fresh_dst_k = fresh_budget - fresh_consumed + fresh_left.log_delta().min(fresh_right.log_delta());

                        let mut fresh_control = alloc_ct(&phase5b_params, module, fresh_dst_k);
                        module
                            .ckks_mul_into(
                                &mut fresh_control,
                                &fresh_left,
                                &fresh_right,
                                &tensor_prepared,
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();

                        let mut fresh_rot_mul = alloc_ct(&phase5b_params, module, fresh_dst_k);
                        module
                            .ckks_mul_into(
                                &mut fresh_rot_mul,
                                &fresh_left,
                                &fresh_rot,
                                &tensor_prepared,
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();

                        let (fresh_ctl_re, fresh_ctl_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &fresh_control,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );
                        let (fresh_rot_out_re, fresh_rot_out_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &fresh_rot_mul,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut fresh_ctl_err = 0.0f64;
                        let mut fresh_rot_err = 0.0f64;
                        let mut fresh_ctl_max_abs = 0.0f64;
                        let mut fresh_rot_max_abs = 0.0f64;

                        for p in 0..m {
                            let lr = left_re[p].to_f64().unwrap();
                            let li = left_im[p].to_f64().unwrap();
                            let rr = right_re[p].to_f64().unwrap();
                            let ri = right_im[p].to_f64().unwrap();
                            let zr = rot_re[p].to_f64().unwrap();
                            let zi = rot_im[p].to_f64().unwrap();

                            let ctl_pr = lr * rr - li * ri;
                            let ctl_pi = lr * ri + li * rr;
                            let rot_pr = lr * zr - li * zi;
                            let rot_pi = lr * zi + li * zr;

                            let fcr = fresh_ctl_re[p].to_f64().unwrap();
                            let fci = fresh_ctl_im[p].to_f64().unwrap();
                            let frr = fresh_rot_out_re[p].to_f64().unwrap();
                            let fri = fresh_rot_out_im[p].to_f64().unwrap();

                            fresh_ctl_err = fresh_ctl_err.max((fcr - ctl_pr).abs()).max((fci - ctl_pi).abs());
                            fresh_rot_err = fresh_rot_err.max((frr - rot_pr).abs()).max((fri - rot_pi).abs());

                            fresh_ctl_max_abs = fresh_ctl_max_abs.max(fcr.abs()).max(fci.abs());
                            fresh_rot_max_abs = fresh_rot_max_abs.max(frr.abs()).max(fri.abs());
                        }

                        println!(
                            "[sheared phase5c2b width8-fresh-reencrypt] backend={} fresh_k={} fresh_budget={} fresh_dst_k={} fresh_control_result_k={} fresh_control_budget={} fresh_rot_result_k={} fresh_rot_budget={} fresh_control_err={:.3e} fresh_rot_err={:.3e} max_abs_fresh_control={:.3e} max_abs_fresh_rot={:.3e}",
                            std::any::type_name::<BE>(),
                            fresh_k,
                            fresh_left.log_budget(),
                            fresh_dst_k,
                            fresh_control.k().as_usize(),
                            fresh_control.log_budget(),
                            fresh_rot_mul.k().as_usize(),
                            fresh_rot_mul.log_budget(),
                            fresh_ctl_err,
                            fresh_rot_err,
                            fresh_ctl_max_abs,
                            fresh_rot_max_abs,
                        );

                        // DECISIVE PROBE: tensor-level comparison at the REAL
                        // cnv_offset (max(a_k,b_k), res_offset = 0).  The earlier
                        // c=30 probe only shows that both operands behave the same
                        // way; it cannot see a common-mode error.  Here the fresh
                        // product is known-good end-to-end, so any difference in
                        // the decrypted tensors localises the fault to
                        // glwe_tensor_apply rather than to relinearization.
                        {
                            use poulpy_hal::layouts::ZnxView as _;
                            let tensor_k = left.k().max(fresh_right.k());
                            let cnv_offset = tensor_k.as_usize();
                            let tensor_layout = poulpy_core::layouts::GLWELayout {
                                n: left.n(),
                                base2k: left.base2k(),
                                k: tensor_k,
                                rank: left.rank(),
                            };
                            let mut hist_tensor = module.glwe_tensor_alloc_from_infos(&tensor_layout);
                            let mut fresh_tensor = module.glwe_tensor_alloc_from_infos(&tensor_layout);
                            module.glwe_tensor_apply(
                                cnv_offset,
                                &mut hist_tensor,
                                left,
                                &fresh_right,
                                &mut tree_scratch.borrow(),
                            );
                            module.glwe_tensor_apply(
                                cnv_offset,
                                &mut fresh_tensor,
                                &fresh_left,
                                &fresh_right,
                                &mut tree_scratch.borrow(),
                            );

                            let mut probe_sk_tensor = module.glwe_secret_tensor_alloc(left.rank());
                            module.glwe_secret_tensor_prepare(&mut probe_sk_tensor, &tree_sk, &mut tree_scratch.borrow());
                            let mut probe_sk_tensor_prepared = module.glwe_secret_tensor_prepared_alloc(left.rank());
                            module.glwe_secret_tensor_prepared_prepare(&mut probe_sk_tensor_prepared, &probe_sk_tensor);

                            let mut hist_tensor_pt = module.glwe_plaintext_alloc_from_infos(&tensor_layout);
                            let mut fresh_tensor_pt = module.glwe_plaintext_alloc_from_infos(&tensor_layout);
                            module.glwe_tensor_decrypt(
                                &hist_tensor,
                                &mut hist_tensor_pt,
                                &phase5b_sk,
                                &probe_sk_tensor_prepared,
                                &mut tree_scratch.borrow(),
                            );
                            module.glwe_tensor_decrypt(
                                &fresh_tensor,
                                &mut fresh_tensor_pt,
                                &phase5b_sk,
                                &probe_sk_tensor_prepared,
                                &mut tree_scratch.borrow(),
                            );

                            // Operand limb structure at the same k the multiply uses.
                            {
                                let dump = |label: &str, ct: &poulpy_core::layouts::GLWE<BE::OwnedBuf, BE::ZnxWord>| {
                                    let r = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(ct);
                                    let d = r.data();
                                    let sz = r.size();
                                    let mut lmax = Vec::with_capacity(sz);
                                    for limb in 0..sz {
                                        let mut m = 0i128;
                                        for &x in d.at(0, limb).iter() {
                                            m = m.max((x as i128).abs());
                                        }
                                        lmax.push(m);
                                    }
                                    let pad = (19usize - (tensor_k.as_usize() % 19)) % 19;
                                    let last = sz.saturating_sub(1);
                                    let notmult = if pad == 0 {
                                        0usize
                                    } else {
                                        let mask = (1i64 << pad) - 1;
                                        d.at(0, last).iter().filter(|&&x| (x & mask) != 0).count()
                                    };
                                    println!(
                                        "[diag operand] backend={} label={} k={} size={} limb_max={:?} pad={} last_limb_not_mult_of_2^{}={}",
                                        std::any::type_name::<BE>(),
                                        label,
                                        tensor_k.as_usize(),
                                        sz,
                                        lmax,
                                        pad,
                                        pad,
                                        notmult,
                                    );
                                };
                                dump("hist-left", left);
                                dump("fresh-left", &fresh_left);
                                dump("fresh-right", &fresh_right);
                            }

                            let hist_ref = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&hist_tensor_pt);
                            let fresh_ref = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&fresh_tensor_pt);
                            let hist_data = hist_ref.data();
                            let fresh_data = fresh_ref.data();
                            let size = hist_ref.size();
                            let mut hist_limb_max = Vec::with_capacity(size);
                            let mut fresh_limb_max = Vec::with_capacity(size);
                            let mut diff_limb_max = Vec::with_capacity(size);
                            let mut diff_limb_nonzero = Vec::with_capacity(size);
                            for limb in 0..size {
                                let (mut hm, mut fm, mut dm, mut dnz) = (0i128, 0i128, 0i128, 0usize);
                                for (&x, &y) in hist_data.at(0, limb).iter().zip(fresh_data.at(0, limb).iter()) {
                                    hm = hm.max((x as i128).abs());
                                    fm = fm.max((y as i128).abs());
                                    let d = (x as i128) - (y as i128);
                                    dm = dm.max(d.abs());
                                    if d != 0 {
                                        dnz += 1;
                                    }
                                }
                                hist_limb_max.push(hm);
                                fresh_limb_max.push(fm);
                                diff_limb_max.push(dm);
                                diff_limb_nonzero.push(dnz);
                            }
                            println!(
                                "[diag tensor c176] backend={} tensor_k={} limbs={} hist_limb_max={:?} fresh_limb_max={:?}",
                                std::any::type_name::<BE>(),
                                tensor_k.as_usize(),
                                size,
                                hist_limb_max,
                                fresh_limb_max,
                            );
                            println!(
                                "[diag tensor c176-diff] backend={} diff_limb_max={:?} diff_limb_nonzero={:?}",
                                std::any::type_name::<BE>(),
                                diff_limb_max,
                                diff_limb_nonzero,
                            );
                            // Limb-wise signed diff of the two outputs (hist vs fresh):
                            // shows exactly which limbs are corrupted and by how much.
                            {
                                let mut outs = [
                                    alloc_ct(&phase5b_params, module, fresh_dst_k),
                                    alloc_ct(&phase5b_params, module, fresh_dst_k),
                                ];
                                for (idx, lct) in [left, &fresh_left].into_iter().enumerate() {
                                    module
                                        .ckks_mul_into(
                                            &mut outs[idx],
                                            lct,
                                            &fresh_right,
                                            &tensor_prepared,
                                            &mut tree_scratch.borrow(),
                                        )
                                        .unwrap();
                                }
                                let pl = poulpy_core::layouts::GLWELayout {
                                    n: left.n(),
                                    base2k: left.base2k(),
                                    k: left.k(),
                                    rank: left.rank(),
                                };
                                let mut pts: Vec<i128> = Vec::new();
                                let mut size = 0usize;
                                for oct in outs.iter() {
                                    let mut op = module.glwe_plaintext_alloc_from_infos(&pl);
                                    module.glwe_decrypt(oct, &mut op, &phase5b_sk, &mut tree_scratch.borrow());
                                    let orf = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&op);
                                    let od = orf.data();
                                    size = orf.size();
                                    for limb in 0..size {
                                        let mut mx = 0i128;
                                        let mut nz = 0usize;
                                        for &x in od.at(0, limb).iter() {
                                            mx = mx.max((x as i128).abs());
                                            if x != 0 {
                                                nz += 1;
                                            }
                                        }
                                        pts.push(mx);
                                        pts.push(nz as i128);
                                    }
                                }
                                println!(
                                    "[diag out-hist] backend={} size={} limb_max_and_nz={:?}",
                                    std::any::type_name::<BE>(),
                                    size,
                                    &pts[..2 * size]
                                );
                                println!(
                                    "[diag out-fresh] backend={} size={} limb_max_and_nz={:?}",
                                    std::any::type_name::<BE>(),
                                    size,
                                    &pts[2 * size..]
                                );
                            }

                            // What the decoder actually sees: decrypt into a k=127
                            // plaintext, i.e. exactly the window ckks_decrypt_decode uses.
                            {
                                let dec_k: usize = 127;
                                let pl127 = poulpy_core::layouts::GLWELayout {
                                    n: left.n(),
                                    base2k: left.base2k(),
                                    k: poulpy_core::layouts::TorusPrecision(dec_k as u32),
                                    rank: left.rank(),
                                };
                                for (mlabel, lct) in [("hist-in", left), ("fresh-in", &fresh_left)] {
                                    let mut out = alloc_ct(&phase5b_params, module, fresh_dst_k);
                                    module
                                        .ckks_mul_into(&mut out, lct, &fresh_right, &tensor_prepared, &mut tree_scratch.borrow())
                                        .unwrap();
                                    for (olabel, oct) in [("in", lct), ("out", &out)] {
                                        let mut op = module.glwe_plaintext_alloc_from_infos(&pl127);
                                        module.glwe_decrypt(oct, &mut op, &phase5b_sk, &mut tree_scratch.borrow());
                                        let orf = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&op);
                                        let od = orf.data();
                                        let osz = orf.size();
                                        let mut olm: Vec<i128> = Vec::with_capacity(osz);
                                        for limb in 0..osz {
                                            let mut mx = 0i128;
                                            for &x in od.at(0, limb).iter() {
                                                mx = mx.max((x as i128).abs());
                                            }
                                            olm.push(mx);
                                        }
                                        println!(
                                            "[diag k127] backend={} ct={} which={} size={} limb_max={:?}",
                                            std::any::type_name::<BE>(),
                                            mlabel,
                                            olabel,
                                            osz,
                                            olm
                                        );
                                    }
                                }
                            }

                            // Real-path output plaintext (tensor + relin via ckks_mul_into).
                            let rp_layout_probe = poulpy_core::layouts::GLWELayout {
                                n: left.n(),
                                base2k: left.base2k(),
                                k: poulpy_core::layouts::TorusPrecision(
                                    ((fresh_dst_k) as u32).min(left.log_budget().min(fresh_right.log_budget()) as u32),
                                ),
                                rank: left.rank(),
                            };
                            {
                                for (mlabel, lct) in [("hist", left), ("fresh", &fresh_left)] {
                                    let mut out = alloc_ct(&phase5b_params, module, fresh_dst_k);
                                    module
                                        .ckks_mul_into(&mut out, lct, &fresh_right, &tensor_prepared, &mut tree_scratch.borrow())
                                        .unwrap();
                                    let mut op = module.glwe_plaintext_alloc_from_infos(&rp_layout_probe);
                                    module.glwe_decrypt(&out, &mut op, &phase5b_sk, &mut tree_scratch.borrow());
                                    let orf = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&op);
                                    let od = orf.data();
                                    let osz = orf.size();
                                    let mut olm: Vec<i128> = Vec::with_capacity(osz);
                                    for limb in 0..osz {
                                        let mut m = 0i128;
                                        for &x in od.at(0, limb).iter() {
                                            m = m.max((x as i128).abs());
                                        }
                                        olm.push(m);
                                    }
                                    println!(
                                        "[diag mul-plaintext] backend={} label={} out_k={} out_size={} limb_max={:?}",
                                        std::any::type_name::<BE>(),
                                        mlabel,
                                        out.k().as_usize(),
                                        osz,
                                        olm
                                    );
                                    let (dre, dim) = ckks_decrypt_decode::<BE, F, E>(
                                        &phase5b_params,
                                        module,
                                        &encoder,
                                        &out,
                                        &phase5b_sk,
                                        &mut tree_scratch.borrow(),
                                    );
                                    let mut mx = 0.0f64;
                                    let mut s0 = (0.0f64, 0.0f64);
                                    for p in 0..m {
                                        let r = dre[p].to_f64().unwrap();
                                        let i = dim[p].to_f64().unwrap();
                                        mx = mx.max(r.abs()).max(i.abs());
                                        if p < 4 {
                                            println!("[diag mul-decode] label={} p={} re={:.6e} im={:.6e}", mlabel, p, r, i);
                                        }
                                    }
                                    println!(
                                        "[diag mul-decode] backend={} label={} max_abs={:.6e}",
                                        std::any::type_name::<BE>(),
                                        mlabel,
                                        mx
                                    );
                                }
                            }

                            // ==== STAGE-BY-STAGE LOCALIZATION OF THE PERTURBATION ====
                            // For a few (limb, delta) perturbations, look at the raw tensor
                            // plaintext and the raw relinearized plaintext: does the injected
                            // content survive the tensor's hi-limb drop, the relinearity, or
                            // does it only appear in the extraction/decode?
                            {
                                use poulpy_hal::layouts::ZnxViewMut as _;
                                let mut sk_tensor = module.glwe_secret_tensor_alloc(left.rank());
                                module.glwe_secret_tensor_prepare(&mut sk_tensor, &tree_sk, &mut tree_scratch.borrow());
                                let mut sk_tensor_prep = module.glwe_secret_tensor_prepared_alloc(left.rank());
                                module.glwe_secret_tensor_prepared_prepare(&mut sk_tensor_prep, &sk_tensor);
                                let out_layout = poulpy_core::layouts::GLWELayout {
                                    n: left.n(),
                                    base2k: left.base2k(),
                                    k: poulpy_core::layouts::TorusPrecision(
                                        (left.log_budget().min(fresh_right.log_budget())) as u32,
                                    ),
                                    rank: left.rank(),
                                };
                                for (limb, delta) in [(0usize, 0i64), (0, 1), (0, 2), (0, 4), (1, 1), (1, 2), (8, 1), (8, 2)] {
                                    let mut c0 = alloc_ct(&phase5b_params, module, fresh_k);
                                    module.ckks_copy(&mut c0, &fresh_left, &mut tree_scratch.borrow()).unwrap();
                                    {
                                        let mut cb = poulpy_core::layouts::GLWEToBackendMut::<BE>::to_backend_mut(&mut c0);
                                        for col in 0..2usize {
                                            for v in cb.data_mut().at_mut(col, limb).iter_mut() {
                                                *v = v.wrapping_add(delta);
                                            }
                                        }
                                    }
                                    let tensor_layout = poulpy_core::layouts::GLWELayout {
                                        n: left.n(),
                                        base2k: left.base2k(),
                                        k: fresh_k.into(),
                                        rank: left.rank(),
                                    };
                                    let mut tn = module.glwe_tensor_alloc_from_infos(&tensor_layout);
                                    module.glwe_tensor_apply(fresh_k, &mut tn, &c0, &fresh_right, &mut tree_scratch.borrow());
                                    let mut tpt = module.glwe_plaintext_alloc_from_infos(&tensor_layout);
                                    module.glwe_tensor_decrypt(
                                        &tn,
                                        &mut tpt,
                                        &phase5b_sk,
                                        &sk_tensor_prep,
                                        &mut tree_scratch.borrow(),
                                    );
                                    let mut rl = module.glwe_alloc_from_infos(&out_layout);
                                    module.glwe_tensor_relinearize(&mut rl, &tn, &tensor_prepared, &mut tree_scratch.borrow());
                                    let mut rpt = module.glwe_plaintext_alloc_from_infos(&out_layout);
                                    module.glwe_decrypt(&rl, &mut rpt, &phase5b_sk, &mut tree_scratch.borrow());
                                    let tl: String = {
                                        let b = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&tpt);
                                        (0..b.size())
                                            .map(|l| b.data().at(0, l).iter().map(|&x| (x as i128).abs()).max().unwrap_or(0))
                                            .map(|m| format!("{}", m))
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    };
                                    let rlm: String = {
                                        let b = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&rpt);
                                        (0..b.size())
                                            .map(|l| b.data().at(0, l).iter().map(|&x| (x as i128).abs()).max().unwrap_or(0))
                                            .map(|m| format!("{}", m))
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    };
                                    let tl_raw: String = {
                                        let b = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&tn);
                                        (0..b.size())
                                            .map(|l| b.data().at(0, l).iter().map(|&x| (x as i128).abs()).max().unwrap_or(0))
                                            .map(|m| format!("{}", m))
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    };
                                    println!(
                                        "[diag perturb-stage-raw] backend={} limb={} delta={} tensor_raw_limbmax=[{}]",
                                        std::any::type_name::<BE>(),
                                        limb,
                                        delta,
                                        tl_raw,
                                    );
                                    println!(
                                        "[diag perturb-stage] backend={} limb={} delta={} tensor_pt_limbmax=[{}] relin_pt_limbmax=[{}]",
                                        std::any::type_name::<BE>(),
                                        limb,
                                        delta,
                                        tl,
                                        rlm,
                                    );
                                }
                            }

                            // ============ CONTROLLED-NOISE REGRESSION ============
                            // Start from two zero-noise fresh ciphertexts (which multiply
                            // correctly) and inject a known offset into one low-significance
                            // limb of the left operand.  If the product explodes, the fault
                            // is the handling of low-limb content, reproducible without SHIP.
                            {
                                use poulpy_hal::layouts::ZnxViewMut as _;
                                // reference: unperturbed fresh x fresh
                                let mut reference: Vec<f64> = Vec::new();
                                {
                                    let mut out = alloc_ct(&phase5b_params, module, fresh_dst_k);
                                    module
                                        .ckks_mul_into(
                                            &mut out,
                                            &fresh_left,
                                            &fresh_right,
                                            &tensor_prepared,
                                            &mut tree_scratch.borrow(),
                                        )
                                        .unwrap();
                                    let (re, im) = ckks_decrypt_decode::<BE, F, E>(
                                        &phase5b_params,
                                        module,
                                        &encoder,
                                        &out,
                                        &phase5b_sk,
                                        &mut tree_scratch.borrow(),
                                    );
                                    for p in 0..m {
                                        reference.push(re[p].to_f64().unwrap());
                                        reference.push(im[p].to_f64().unwrap());
                                    }
                                }
                                let n = module.n();
                                for limb in [0usize] {
                                    for delta in [
                                        1i64,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        15,
                                        17,
                                        31,
                                        32,
                                        33,
                                        63,
                                        64,
                                        65,
                                        127,
                                        128,
                                        129,
                                        255,
                                        257,
                                        511,
                                        513,
                                        1023,
                                        1025,
                                        1 << 11,
                                        (1 << 11) + 1,
                                        1 << 13,
                                        (1 << 13) + 1,
                                        1 << 15,
                                        (1 << 15) + 1,
                                        1 << 16,
                                        (1 << 16) + 1,
                                        1 << 17,
                                    ] {
                                        let mut c0 = alloc_ct(&phase5b_params, module, fresh_k);
                                        module.ckks_copy(&mut c0, &fresh_left, &mut tree_scratch.borrow()).unwrap();
                                        {
                                            let mut cb = poulpy_core::layouts::GLWEToBackendMut::<BE>::to_backend_mut(&mut c0);
                                            for col in 0..2usize {
                                                for x in cb.data_mut().at_mut(col, limb).iter_mut() {
                                                    *x = x.wrapping_add(delta);
                                                }
                                            }
                                        }
                                        let mut out = alloc_ct(&phase5b_params, module, fresh_dst_k);
                                        module
                                            .ckks_mul_into(
                                                &mut out,
                                                &c0,
                                                &fresh_right,
                                                &tensor_prepared,
                                                &mut tree_scratch.borrow(),
                                            )
                                            .unwrap();
                                        let (re, im) = ckks_decrypt_decode::<BE, F, E>(
                                            &phase5b_params,
                                            module,
                                            &encoder,
                                            &out,
                                            &phase5b_sk,
                                            &mut tree_scratch.borrow(),
                                        );
                                        let mut worst = 0.0f64;
                                        let mut mx = 0.0f64;
                                        for p in 0..m {
                                            let r = re[p].to_f64().unwrap();
                                            let i = im[p].to_f64().unwrap();
                                            mx = mx.max(r.abs()).max(i.abs());
                                            worst = worst.max((r - reference[2 * p]).abs()).max((i - reference[2 * p + 1]).abs());
                                        }
                                        println!(
                                            "[diag noise-sweep] limb={} delta={} max_abs={:.3e} err={:.3e}",
                                            limb, delta, mx, worst,
                                        );
                                    }
                                }
                            }

                            // ================= DECISIVE SPLIT 1 =================
                            // Does the relinearization preserve the tensor's plaintext?
                            // Compare, per coefficient, the raw tensor plaintext (as
                            // decrypted by glwe_tensor_decrypt) against the raw
                            // relinearized GLWE plaintext, converting the former with an
                            // exact host-side signed shift.  No CKKS extraction/decode is
                            // involved, so a mismatch here convicts
                            // gglwe_product_dft / the FFT64 VMP.
                            {
                                fn digits(limbs: &[i128]) -> Vec<i128> {
                                    let m: i128 = 1 << 19;
                                    let mut d = limbs.to_vec();
                                    for j in (1..d.len()).rev() {
                                        if d[j] < 0 {
                                            let q = (-d[j] + m - 1) / m;
                                            d[j] += q * m;
                                            d[j - 1] -= q;
                                        }
                                    }
                                    d
                                }
                                // digits of round(A >> s) with n_out digits, A = sum d[j] 2^(19*(n-1-j)),
                                // ties toward +inf (same rule as normalize_integer_oracle).
                                fn rsh(d: &[i128], s: usize, n_out: usize) -> Vec<i128> {
                                    let n = d.len();
                                    let mut a = d.to_vec();
                                    if s > 0 {
                                        let r = s - 1;
                                        let (steps, rem) = (r / 19, r % 19);
                                        let mut carry: i128 = 1 << rem;
                                        let mut j = n as i64 - 1 - steps as i64;
                                        while carry != 0 && j >= 0 {
                                            let v = a[j as usize] + carry;
                                            a[j as usize] = v & ((1 << 19) - 1);
                                            carry = v >> 19;
                                            j -= 1;
                                        }
                                    }
                                    let (steps, rem) = (s / 19, s % 19);
                                    let mut out = vec![0i128; n_out];
                                    for j in 0..n_out {
                                        let src = j + steps;
                                        if src >= n {
                                            break;
                                        }
                                        let hi = a[src] >> rem;
                                        let lo = if rem > 0 && src > 0 {
                                            (a[src - 1] & ((1 << rem) - 1)) << (19 - rem)
                                        } else {
                                            0
                                        };
                                        out[j] = hi | lo;
                                    }
                                    out
                                }

                                let mut sk_tensor = module.glwe_secret_tensor_alloc(left.rank());
                                module.glwe_secret_tensor_prepare(&mut sk_tensor, &tree_sk, &mut tree_scratch.borrow());
                                let mut sk_tensor_prep = module.glwe_secret_tensor_prepared_alloc(left.rank());
                                module.glwe_secret_tensor_prepared_prepare(&mut sk_tensor_prep, &sk_tensor);

                                let out_layout = poulpy_core::layouts::GLWELayout {
                                    n: left.n(),
                                    base2k: left.base2k(),
                                    k: poulpy_core::layouts::TorusPrecision(
                                        (left.log_budget().min(fresh_right.log_budget())) as u32,
                                    ),
                                    rank: left.rank(),
                                };
                                for (label, tct) in [("hist", &hist_tensor), ("fresh", &fresh_tensor)] {
                                    let mut tpt = module.glwe_plaintext_alloc_from_infos(&tensor_layout);
                                    module.glwe_tensor_decrypt(
                                        tct,
                                        &mut tpt,
                                        &phase5b_sk,
                                        &sk_tensor_prep,
                                        &mut tree_scratch.borrow(),
                                    );
                                    let mut rl = module.glwe_alloc_from_infos(&out_layout);
                                    module.glwe_tensor_relinearize(&mut rl, tct, &tensor_prepared, &mut tree_scratch.borrow());
                                    let mut rpt = module.glwe_plaintext_alloc_from_infos(&out_layout);
                                    module.glwe_decrypt(&rl, &mut rpt, &phase5b_sk, &mut tree_scratch.borrow());

                                    let tb = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&tpt);
                                    let rb = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&rpt);
                                    let td = tb.data();
                                    let rd = rb.data();
                                    let tsz = tb.size();
                                    let rsz = rb.size();
                                    let t_k = tpt.k().as_usize();
                                    let r_k = rpt.k().as_usize();
                                    let shift = 19 * (tsz - rsz) + (t_k - r_k);

                                    // Calibrate the shift: for each candidate shift find the
                                    // most significant digit index at which the two sides stop
                                    // agreeing (allowing the +/-1 rounding ulp).  The true
                                    // shift maximises that index.  If hist and fresh need
                                    // different shifts, the relinearization is not uniform.
                                    let mut best_shift: i64 = -1;
                                    let mut best_top: i64 = -1;
                                    let mut best_nbad = 0usize;
                                    let mut scan: Vec<i64> = Vec::new();
                                    for cand in (0..=152i64).step_by(1) {
                                        let mut top_bad: i64 = rsz as i64;
                                        let mut nbad = 0usize;
                                        for i in 0..module.n() {
                                            let tl: Vec<i128> = (0..tsz).map(|l| td.at(0, l)[i] as i128).collect();
                                            let got: Vec<i128> = (0..rsz).map(|l| rd.at(0, l)[i] as i128).collect();
                                            let exp = rsh(&digits(&tl), cand as usize, rsz);
                                            let mut e = exp.clone();
                                            for j in (1..rsz).rev() {
                                                if e[j] > (1 << 18) {
                                                    e[j] -= 1 << 19;
                                                    e[j - 1] += 1;
                                                }
                                            }
                                            for j in 0..rsz {
                                                if (e[j] - got[j]).abs() > 1 {
                                                    nbad += 1;
                                                    if (j as i64) < top_bad {
                                                        top_bad = j as i64;
                                                    }
                                                }
                                            }
                                        }
                                        if top_bad > best_top {
                                            best_top = top_bad;
                                            best_shift = cand;
                                            best_nbad = nbad;
                                        }
                                        if cand % 8 == 0 {
                                            scan.push(top_bad);
                                        }
                                    }
                                    println!(
                                        "[diag relin-split] backend={} label={} tensor_k={} tsize={} relin_k={} rsize={} shift_guess={} best_shift={} best_top_bad_digit={} best_nbad={} scan={:?}",
                                        std::any::type_name::<BE>(),
                                        label,
                                        t_k,
                                        tsz,
                                        r_k,
                                        rsz,
                                        shift,
                                        best_shift,
                                        best_top,
                                        best_nbad,
                                        scan,
                                    );
                                }
                            }

                            // Relinearization output plaintext at the same tensors.
                            let rp_layout_probe = poulpy_core::layouts::GLWELayout {
                                n: left.n(),
                                base2k: left.base2k(),
                                k: poulpy_core::layouts::TorusPrecision((left.log_budget().min(fresh_right.log_budget())) as u32),
                                rank: left.rank(),
                            };
                            {
                                let out_layout = poulpy_core::layouts::GLWELayout {
                                    n: left.n(),
                                    base2k: left.base2k(),
                                    k: poulpy_core::layouts::TorusPrecision(
                                        (left.log_budget().min(fresh_right.log_budget())) as u32,
                                    ),
                                    rank: left.rank(),
                                };
                                for (rlabel, tct) in [("hist", &hist_tensor), ("fresh", &fresh_tensor)] {
                                    let mut rl = module.glwe_alloc_from_infos(&out_layout);
                                    module.glwe_tensor_relinearize(&mut rl, tct, &tensor_prepared, &mut tree_scratch.borrow());
                                    let mut rp = module.glwe_plaintext_alloc_from_infos(&rp_layout_probe);
                                    module.glwe_decrypt(&rl, &mut rp, &phase5b_sk, &mut tree_scratch.borrow());
                                    let rr = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&rp);
                                    let rd = rr.data();
                                    let rsz = rr.size();
                                    let mut rlm: Vec<i128> = Vec::with_capacity(rsz);
                                    for limb in 0..rsz {
                                        let mut m = 0i128;
                                        for &x in rd.at(0, limb).iter() {
                                            m = m.max((x as i128).abs());
                                        }
                                        rlm.push(m);
                                    }
                                    println!(
                                        "[diag relin-plaintext] backend={} label={} out_k={} out_size={} limb_max={:?}",
                                        std::any::type_name::<BE>(),
                                        rlabel,
                                        rl.k().as_usize(),
                                        rsz,
                                        rlm
                                    );
                                }
                            }

                            // Operand plaintext limb structure at the same k.
                            {
                                let pl = poulpy_core::layouts::GLWELayout {
                                    n: left.n(),
                                    base2k: left.base2k(),
                                    k: left.k(),
                                    rank: left.rank(),
                                };
                                for (olabel, oct) in [
                                    ("hist-left", left),
                                    ("fresh-left", &fresh_left),
                                    ("fresh-right", &fresh_right),
                                ] {
                                    let mut opt = module.glwe_plaintext_alloc_from_infos(&pl);
                                    module.glwe_decrypt(oct, &mut opt, &phase5b_sk, &mut tree_scratch.borrow());
                                    let opr = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&opt);
                                    let od = opr.data();
                                    let osz = opr.size();
                                    let mut olm: Vec<i128> = Vec::with_capacity(osz);
                                    for limb in 0..osz {
                                        let mut m = 0i128;
                                        for &x in od.at(0, limb).iter() {
                                            m = m.max((x as i128).abs());
                                        }
                                        olm.push(m);
                                    }
                                    println!(
                                        "[diag operand-plaintext] backend={} label={} k={} size={} limb_max={:?}",
                                        std::any::type_name::<BE>(),
                                        olabel,
                                        left.k().as_usize(),
                                        osz,
                                        olm
                                    );
                                }
                            }

                            for (tlabel, tct) in [("hist", &hist_tensor), ("fresh", &fresh_tensor)] {
                                let tb = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(tct);
                                let td = tb.data();
                                let tsz = tb.size();
                                let tcols = 3usize;
                                let mut per_col: Vec<Vec<i128>> = Vec::new();
                                for c in 0..tcols {
                                    let mut lm: Vec<i128> = Vec::with_capacity(tsz);
                                    for limb in 0..tsz {
                                        let mut m = 0i128;
                                        for &x in td.at(c, limb).iter() {
                                            m = m.max((x as i128).abs());
                                        }
                                        lm.push(m);
                                    }
                                    per_col.push(lm);
                                }
                                println!(
                                    "[diag tensor-raw] backend={} label={} cols={} size={} limb_max_per_col={:?}",
                                    std::any::type_name::<BE>(),
                                    tlabel,
                                    tcols,
                                    tsz,
                                    per_col
                                );
                            }

                            // SCHOOLBOOK REFERENCE against the raw DFT kernel output:
                            // run cnv_apply_dft + IDFT by hand on one tensor column
                            // and compare the big-domain coefficients against an
                            // exact i64 schoolbook convolution of the operand limbs.
                            {
                                use poulpy_hal::api::{VecZnxBigAlloc, VecZnxDftAlloc};
                                let n = module.n();
                                let base2k = left.base2k().as_usize();
                                let a_k = left.k().as_usize();
                                let b_k = fresh_right.k().as_usize();
                                let a_eff = a_k.div_ceil(base2k);
                                let b_eff = b_k.div_ceil(base2k);
                                let (hi, lo) = poulpy_core::cnv_offset_to_limb_offset(cnv_offset, base2k);
                                let lo_u = lo.unsigned_abs() as usize;
                                let res_size = size;
                                let diag = (a_eff + b_eff - hi).min((res_size * base2k + lo_u).div_ceil(base2k));
                                let mk = |k: usize| -> i64 {
                                    if k % base2k == 0 {
                                        !0i64
                                    } else {
                                        (!0i64) << (base2k - k % base2k)
                                    }
                                };
                                let cols = 2usize;
                                let lhs_b = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(left);
                                let rhs_b = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&fresh_right);
                                let hist_left_d = lhs_b.data();
                                let fresh_right_d = rhs_b.data();
                                println!(
                                    "[diag schoolboy] backend={} a_k={} b_k={} a_eff={} b_eff={} hi={} lo={} diag={} res_size={}",
                                    std::any::type_name::<BE>(),
                                    a_k,
                                    b_k,
                                    a_eff,
                                    b_eff,
                                    hi,
                                    lo,
                                    diag,
                                    res_size,
                                );
                                let fresh_left_b = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(&fresh_left);
                                let fresh_left_d = fresh_left_b.data();
                                for (label, lcols, rcols) in
                                    [("hist", hist_left_d, fresh_right_d), ("fresh", fresh_left_d, fresh_right_d)]
                                {
                                    let mut a_prep = module.cnv_pvec_left_alloc(cols, a_eff);
                                    let mut b_prep = module.cnv_pvec_right_alloc(cols, b_eff);
                                    {
                                        let mut ap = poulpy_hal::layouts::CnvPVecLToBackendMut::<BE>::to_backend_mut(&mut a_prep);
                                        module.cnv_prepare_left(&mut ap, &lcols, mk(a_k), &mut tree_scratch.borrow());
                                    }
                                    {
                                        let mut bp = poulpy_hal::layouts::CnvPVecRToBackendMut::<BE>::to_backend_mut(&mut b_prep);
                                        module.cnv_prepare_right(&mut bp, &rcols, mk(b_k), &mut tree_scratch.borrow());
                                    }
                                    let a_prep_ref = poulpy_hal::layouts::CnvPVecLToBackendRef::<BE>::to_backend_ref(&a_prep);
                                    let b_prep_ref = poulpy_hal::layouts::CnvPVecRToBackendRef::<BE>::to_backend_ref(&b_prep);
                                    let mut res_dft = module.vec_znx_dft_alloc(1, diag);
                                    {
                                        let mut rd =
                                            poulpy_hal::layouts::VecZnxDftToBackendMut::<BE>::to_backend_mut(&mut res_dft);
                                        module.cnv_apply_dft(
                                            hi,
                                            &mut rd,
                                            0,
                                            &a_prep_ref,
                                            0,
                                            &b_prep_ref,
                                            0,
                                            &mut tree_scratch.borrow(),
                                        );
                                    }
                                    let mut res_big = module.vec_znx_big_alloc(1, diag);
                                    {
                                        let mut rb =
                                            poulpy_hal::layouts::VecZnxBigToBackendMut::<BE>::to_backend_mut(&mut res_big);
                                        let mut rd =
                                            poulpy_hal::layouts::VecZnxDftToBackendMut::<BE>::to_backend_mut(&mut res_dft);
                                        module.vec_znx_idft_apply_tmpa(&mut rb, 0, &mut rd, 0);
                                    }
                                    // exact schoolbook reference on the same limb window
                                    let mut ref_limbs: Vec<Vec<i64>> = (0..diag).map(|_| vec![0i64; n]).collect();
                                    for pp in 0..a_eff {
                                        for qq in 0..b_eff {
                                            let idx = pp + qq;
                                            if idx < hi || idx - hi >= diag {
                                                continue;
                                            }
                                            let al = lcols.at(0, pp);
                                            let bl = rcols.at(0, qq);
                                            let out = &mut ref_limbs[idx - hi];
                                            for i in 0..n {
                                                let ai = al[i] as i128;
                                                if ai == 0 {
                                                    continue;
                                                }
                                                for j in 0..n {
                                                    let v = ai * (bl[j] as i128);
                                                    let k = i + j;
                                                    let (kk, sgn) = if k < n { (k, 1i128) } else { (k - n, -1i128) };
                                                    let cur = out[kk] as i128;
                                                    out[kk] = (cur + sgn * v) as i64;
                                                }
                                            }
                                        }
                                    }
                                    let rbr = poulpy_hal::layouts::VecZnxBigToBackendRef::<BE>::to_backend_ref(&res_big);
                                    let mut got_limb_max: Vec<i128> = Vec::with_capacity(diag);
                                    let mut err_limb_max: Vec<i128> = Vec::with_capacity(diag);
                                    let mut nbad: Vec<usize> = Vec::with_capacity(diag);
                                    for limb in 0..diag {
                                        let mut gm = 0i128;
                                        let mut em = 0i128;
                                        let mut nb = 0usize;
                                        for (g, &w) in rbr.at(0, limb).iter().zip(ref_limbs[limb].iter()) {
                                            let g: i128 = format!("{}", g).parse().unwrap();
                                            gm = gm.max(g.abs());
                                            let d = g - (w as i128);
                                            if d != 0 {
                                                nb += 1;
                                                em = em.max(d.abs());
                                            }
                                        }
                                        got_limb_max.push(gm);
                                        err_limb_max.push(em);
                                        nbad.push(nb);
                                    }
                                    println!("[diag schoolboy] label={} got_limb_max={:?}", label, got_limb_max);
                                    println!(
                                        "[diag schoolboy] label={} err_limb_max={:?} nbad={:?}",
                                        label, err_limb_max, nbad
                                    );

                                    // ---- pairwise column (i=0, j=1): (a0 + a1) (x) (b0 + b1)
                                    let mut asum: Vec<Vec<i64>> = Vec::with_capacity(a_eff);
                                    for pp in 0..a_eff {
                                        let mut v = lcols.at(0, pp).to_vec();
                                        for (x, y) in v.iter_mut().zip(lcols.at(1, pp).iter()) {
                                            *x += *y;
                                        }
                                        asum.push(v);
                                    }
                                    let mut bsum: Vec<Vec<i64>> = Vec::with_capacity(b_eff);
                                    for qq in 0..b_eff {
                                        let mut v = rcols.at(0, qq).to_vec();
                                        for (x, y) in v.iter_mut().zip(rcols.at(1, qq).iter()) {
                                            *x += *y;
                                        }
                                        bsum.push(v);
                                    }
                                    let mut ref_pw: Vec<Vec<i64>> = (0..diag).map(|_| vec![0i64; n]).collect();
                                    for pp in 0..a_eff {
                                        for qq in 0..b_eff {
                                            let idx = pp + qq;
                                            if idx < hi || idx - hi >= diag {
                                                continue;
                                            }
                                            let al = &asum[pp];
                                            let bl = &bsum[qq];
                                            let out = &mut ref_pw[idx - hi];
                                            for i in 0..n {
                                                let ai = al[i] as i128;
                                                if ai == 0 {
                                                    continue;
                                                }
                                                for j in 0..n {
                                                    let v = ai * (bl[j] as i128);
                                                    let k = i + j;
                                                    let (kk, sgn) = if k < n { (k, 1i128) } else { (k - n, -1i128) };
                                                    let cur = out[kk] as i128;
                                                    out[kk] = (cur + sgn * v) as i64;
                                                }
                                            }
                                        }
                                    }
                                    {
                                        let mut rd =
                                            poulpy_hal::layouts::VecZnxDftToBackendMut::<BE>::to_backend_mut(&mut res_dft);
                                        module.cnv_pairwise_apply_dft(
                                            hi,
                                            &mut rd,
                                            0,
                                            &a_prep_ref,
                                            &b_prep_ref,
                                            0,
                                            1,
                                            &mut tree_scratch.borrow(),
                                        );
                                    }
                                    let mut res_big2 = module.vec_znx_big_alloc(1, diag);
                                    {
                                        let mut rb =
                                            poulpy_hal::layouts::VecZnxBigToBackendMut::<BE>::to_backend_mut(&mut res_big2);
                                        let mut rd =
                                            poulpy_hal::layouts::VecZnxDftToBackendMut::<BE>::to_backend_mut(&mut res_dft);
                                        module.vec_znx_idft_apply_tmpa(&mut rb, 0, &mut rd, 0);
                                    }
                                    let rbr2 = poulpy_hal::layouts::VecZnxBigToBackendRef::<BE>::to_backend_ref(&res_big2);
                                    let mut pw_err: Vec<i128> = Vec::with_capacity(diag);
                                    let mut pw_nbad: Vec<usize> = Vec::with_capacity(diag);
                                    for limb in 0..diag {
                                        let mut em = 0i128;
                                        let mut nb = 0usize;
                                        for (g, &w) in rbr2.at(0, limb).iter().zip(ref_pw[limb].iter()) {
                                            let g: i128 = format!("{}", g).parse().unwrap();
                                            let d = g - (w as i128);
                                            if d != 0 {
                                                nb += 1;
                                                em = em.max(d.abs());
                                            }
                                        }
                                        pw_err.push(em);
                                        pw_nbad.push(nb);
                                    }
                                    println!(
                                        "[diag schoolboy-pw] label={} err_limb_max={:?} nbad={:?}",
                                        label, pw_err, pw_nbad
                                    );

                                    // ---- diagonal column (i=1, j=1): the mask x mask term
                                    {
                                        let mut rd =
                                            poulpy_hal::layouts::VecZnxDftToBackendMut::<BE>::to_backend_mut(&mut res_dft);
                                        module.cnv_apply_dft(
                                            hi,
                                            &mut rd,
                                            0,
                                            &a_prep_ref,
                                            1,
                                            &b_prep_ref,
                                            1,
                                            &mut tree_scratch.borrow(),
                                        );
                                    }
                                    let mut res_big3 = module.vec_znx_big_alloc(1, diag);
                                    {
                                        let mut rb =
                                            poulpy_hal::layouts::VecZnxBigToBackendMut::<BE>::to_backend_mut(&mut res_big3);
                                        let mut rd =
                                            poulpy_hal::layouts::VecZnxDftToBackendMut::<BE>::to_backend_mut(&mut res_dft);
                                        module.vec_znx_idft_apply_tmpa(&mut rb, 0, &mut rd, 0);
                                    }
                                    let mut ref_11: Vec<Vec<i64>> = (0..diag).map(|_| vec![0i64; n]).collect();
                                    for pp in 0..a_eff {
                                        for qq in 0..b_eff {
                                            let idx = pp + qq;
                                            if idx < hi || idx - hi >= diag {
                                                continue;
                                            }
                                            let al = lcols.at(1, pp);
                                            let bl = rcols.at(1, qq);
                                            let out = &mut ref_11[idx - hi];
                                            for i in 0..n {
                                                let ai = al[i] as i128;
                                                if ai == 0 {
                                                    continue;
                                                }
                                                for j in 0..n {
                                                    let v = ai * (bl[j] as i128);
                                                    let k = i + j;
                                                    let (kk, sgn) = if k < n { (k, 1i128) } else { (k - n, -1i128) };
                                                    let cur = out[kk] as i128;
                                                    out[kk] = (cur + sgn * v) as i64;
                                                }
                                            }
                                        }
                                    }
                                    let rbr3 = poulpy_hal::layouts::VecZnxBigToBackendRef::<BE>::to_backend_ref(&res_big3);
                                    let mut err3: Vec<i128> = Vec::with_capacity(diag);
                                    let mut nbad3: Vec<usize> = Vec::with_capacity(diag);
                                    for limb in 0..diag {
                                        let mut em = 0i128;
                                        let mut nb = 0usize;
                                        for (g, &w) in rbr3.at(0, limb).iter().zip(ref_11[limb].iter()) {
                                            let g: i128 = format!("{}", g).parse().unwrap();
                                            let d = g - (w as i128);
                                            if d != 0 {
                                                nb += 1;
                                                em = em.max(d.abs());
                                            }
                                        }
                                        err3.push(em);
                                        nbad3.push(nb);
                                    }
                                    println!(
                                        "[diag schoolboy-d11] label={} err_limb_max={:?} nbad={:?}",
                                        label, err3, nbad3
                                    );
                                }
                            }
                        }

                        // History/representation probe.  Normalize copies of the
                        // chained width-4 nodes *without* decrypting or re-encrypting
                        // them, then retry the same control multiplication.  This keeps
                        // the RLWE noise/history intact while forcing every limb into
                        // the canonical radix representation.
                        //
                        // Two hybrid products additionally replace only one chained
                        // operand by its fresh encryption.  Together these cases
                        // distinguish:
                        //   - non-canonical limb representation,
                        //   - one bad operand/history,
                        //   - a two-chained-input FFT/convolution stability limit.
                        let mut norm_left = alloc_ct(&phase5b_params, module, fresh_k);
                        module.ckks_copy(&mut norm_left, left, &mut tree_scratch.borrow()).unwrap();
                        module.glwe_normalize_assign_default(&mut norm_left, &mut tree_scratch.borrow());

                        let mut norm_right = alloc_ct(&phase5b_params, module, fresh_k);
                        module.ckks_copy(&mut norm_right, right, &mut tree_scratch.borrow()).unwrap();
                        module.glwe_normalize_assign_default(&mut norm_right, &mut tree_scratch.borrow());

                        assert_eq!(norm_left.k(), left.k());
                        assert_eq!(norm_right.k(), right.k());
                        assert_eq!(norm_left.log_delta(), left.log_delta());
                        assert_eq!(norm_right.log_delta(), right.log_delta());
                        assert_eq!(norm_left.log_budget(), left.log_budget());
                        assert_eq!(norm_right.log_budget(), right.log_budget());

                        let (norm_left_re, norm_left_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &norm_left,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );
                        let (norm_right_re, norm_right_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &norm_right,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut norm_input_err = 0.0f64;
                        for p in 0..m {
                            norm_input_err = norm_input_err
                                .max((norm_left_re[p].to_f64().unwrap() - left_re[p].to_f64().unwrap()).abs())
                                .max((norm_left_im[p].to_f64().unwrap() - left_im[p].to_f64().unwrap()).abs())
                                .max((norm_right_re[p].to_f64().unwrap() - right_re[p].to_f64().unwrap()).abs())
                                .max((norm_right_im[p].to_f64().unwrap() - right_im[p].to_f64().unwrap()).abs());
                        }

                        let mut norm_control = alloc_ct(&phase5b_params, module, fresh_dst_k);
                        module
                            .ckks_mul_into(
                                &mut norm_control,
                                &norm_left,
                                &norm_right,
                                &tensor_prepared,
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();
                        let (norm_ctl_re, norm_ctl_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &norm_control,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut chained_fresh_right = alloc_ct(&phase5b_params, module, fresh_dst_k);
                        module
                            .ckks_mul_into(
                                &mut chained_fresh_right,
                                left,
                                &fresh_right,
                                &tensor_prepared,
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();
                        let (cfr_re, cfr_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &chained_fresh_right,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut fresh_left_chained = alloc_ct(&phase5b_params, module, fresh_dst_k);
                        module
                            .ckks_mul_into(
                                &mut fresh_left_chained,
                                &fresh_left,
                                right,
                                &tensor_prepared,
                                &mut tree_scratch.borrow(),
                            )
                            .unwrap();
                        let (flc_re, flc_im) = ckks_decrypt_decode::<BE, F, E>(
                            &phase5b_params,
                            module,
                            &encoder,
                            &fresh_left_chained,
                            &phase5b_sk,
                            &mut tree_scratch.borrow(),
                        );

                        let mut norm_control_err = 0.0f64;
                        let mut chained_fresh_right_err = 0.0f64;
                        let mut fresh_left_chained_err = 0.0f64;
                        let mut norm_control_max_abs = 0.0f64;
                        let mut chained_fresh_right_max_abs = 0.0f64;
                        let mut fresh_left_chained_max_abs = 0.0f64;

                        for p in 0..m {
                            let lr = left_re[p].to_f64().unwrap();
                            let li = left_im[p].to_f64().unwrap();
                            let rr = right_re[p].to_f64().unwrap();
                            let ri = right_im[p].to_f64().unwrap();
                            let pr = lr * rr - li * ri;
                            let pi = lr * ri + li * rr;

                            let nr = norm_ctl_re[p].to_f64().unwrap();
                            let ni = norm_ctl_im[p].to_f64().unwrap();
                            let ar = cfr_re[p].to_f64().unwrap();
                            let ai = cfr_im[p].to_f64().unwrap();
                            let br = flc_re[p].to_f64().unwrap();
                            let bi = flc_im[p].to_f64().unwrap();

                            norm_control_err = norm_control_err.max((nr - pr).abs()).max((ni - pi).abs());
                            chained_fresh_right_err = chained_fresh_right_err.max((ar - pr).abs()).max((ai - pi).abs());
                            fresh_left_chained_err = fresh_left_chained_err.max((br - pr).abs()).max((bi - pi).abs());

                            norm_control_max_abs = norm_control_max_abs.max(nr.abs()).max(ni.abs());
                            chained_fresh_right_max_abs = chained_fresh_right_max_abs.max(ar.abs()).max(ai.abs());
                            fresh_left_chained_max_abs = fresh_left_chained_max_abs.max(br.abs()).max(bi.abs());
                        }

                        println!(
                            "[sheared phase5c2b width8-history-probe] backend={} norm_input_err={:.3e} norm_control_err={:.3e} chained_x_fresh_err={:.3e} fresh_x_chained_err={:.3e} max_abs_norm_control={:.3e} max_abs_chained_x_fresh={:.3e} max_abs_fresh_x_chained={:.3e}",
                            std::any::type_name::<BE>(),
                            norm_input_err,
                            norm_control_err,
                            chained_fresh_right_err,
                            fresh_left_chained_err,
                            norm_control_max_abs,
                            chained_fresh_right_max_abs,
                            fresh_left_chained_max_abs,
                        );

                        // FFT64 history diagnostic: compare the physical/effective
                        // representation of a chained node with its fresh encryption,
                        // then explicitly truncate the chained node to full base2k
                        // boundaries before one more ciphertext multiplication.
                        //
                        // k=176 with base2k=19 is 9*19+5.  Therefore:
                        //   171 = 9 full limbs (drop only the partial-limb 5 bits),
                        //   152 = 8 full limbs (drop one additional full limb).
                        // If 171 already repairs the product, the remaining fault is
                        // strongly tied to partial-limb handling.  If only 152 repairs
                        // it, the FFT convolution likely needs more dynamic-range
                        // headroom after chained multiplication.
                        println!(
                            "[sheared phase5c2b width8-repr] backend={} base2k={} chained_size={} fresh_size={} chained_k={} fresh_k={} chained_max_k={} fresh_max_k={} chained_delta={} fresh_delta={} chained_budget={} fresh_budget={}",
                            std::any::type_name::<BE>(),
                            left.base2k().as_usize(),
                            left.size(),
                            fresh_left.size(),
                            left.k().as_usize(),
                            fresh_left.k().as_usize(),
                            left.max_k().as_usize(),
                            fresh_left.max_k().as_usize(),
                            left.log_delta(),
                            fresh_left.log_delta(),
                            left.log_budget(),
                            fresh_left.log_budget(),
                        );

                        if left.base2k().as_usize() == 19 && left.k().as_usize() == 176 {
                            for trunc_k in [171usize] {
                                let trunc_delta = left.log_delta();
                                assert!(trunc_k > trunc_delta);
                                let trunc_budget = trunc_k - trunc_delta;
                                // For equal-scale ct*ct multiplication Poulpy's natural
                                // result width is the input log_budget.
                                let trunc_dst_k = trunc_budget;

                                let mut trunc_left = alloc_ct(&phase5b_params, module, trunc_k);
                                trunc_left.set_log_delta(trunc_delta);
                                trunc_left.set_log_budget(trunc_budget);
                                trunc_left.set_log_sparsity(left.log_sparsity());
                                trunc_left.set_slots(left.slots());
                                module.glwe_normalize_default(&mut trunc_left, left, &mut tree_scratch.borrow());

                                let (trunc_left_re, trunc_left_im) = ckks_decrypt_decode::<BE, F, E>(
                                    &phase5b_params,
                                    module,
                                    &encoder,
                                    &trunc_left,
                                    &phase5b_sk,
                                    &mut tree_scratch.borrow(),
                                );

                                // Build the fresh controls by truncating the already-valid
                                // k=176 fresh encryptions.  Do not call ckks_encrypt() at
                                // trunc_k: that helper encodes a high-width plaintext first
                                // and ckks_add_pt_vec correctly rejects aligning that plaintext
                                // to a 171/152-bit ciphertext.  Normalizing from fresh_left /
                                // fresh_right changes only the ciphertext precision and keeps
                                // this probe free of an unrelated plaintext-alignment failure.
                                let mut fresh_trunc_left = alloc_ct(&phase5b_params, module, trunc_k);
                                fresh_trunc_left.set_log_delta(trunc_delta);
                                fresh_trunc_left.set_log_budget(trunc_budget);
                                fresh_trunc_left.set_log_sparsity(fresh_left.log_sparsity());
                                fresh_trunc_left.set_slots(fresh_left.slots());
                                module.glwe_normalize_default(&mut fresh_trunc_left, &fresh_left, &mut tree_scratch.borrow());

                                let mut fresh_trunc_right = alloc_ct(&phase5b_params, module, trunc_k);
                                fresh_trunc_right.set_log_delta(trunc_delta);
                                fresh_trunc_right.set_log_budget(trunc_budget);
                                fresh_trunc_right.set_log_sparsity(fresh_right.log_sparsity());
                                fresh_trunc_right.set_slots(fresh_right.slots());
                                module.glwe_normalize_default(&mut fresh_trunc_right, &fresh_right, &mut tree_scratch.borrow());

                                use poulpy_hal::layouts::ZnxView as _;
                                // Raw radix-limb dynamic-range probe.
                                //
                                // At trunc_k = 171 = 9 * 19 there is no partial
                                // limb. Compare historical and fresh ciphertext
                                // coefficient-limb ranges before FFT convolution.
                                {
                                    let radix_bits = trunc_left.base2k().as_usize();
                                    let radix_half = 1i128 << (radix_bits - 1);
                                    let radix_full = 1i128 << radix_bits;

                                    for (label, ct) in [
                                        ("hist-left", &trunc_left),
                                        ("fresh-left", &fresh_trunc_left),
                                        ("fresh-right", &fresh_trunc_right),
                                    ] {
                                        let ct_ref = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref(ct);
                                        let data = ct_ref.data();

                                        let mut global_max = 0i128;
                                        let mut global_over_half = 0usize;
                                        let mut global_over_full = 0usize;

                                        for col in 0..2usize {
                                            let mut limb_max = Vec::with_capacity(ct.size());
                                            let mut limb_over_half = Vec::with_capacity(ct.size());
                                            let mut limb_over_full = Vec::with_capacity(ct.size());

                                            for limb in 0..ct.size() {
                                                let coeffs = data.at(col, limb);
                                                let mut max_abs = 0i128;
                                                let mut over_half = 0usize;
                                                let mut over_full = 0usize;

                                                for &x in coeffs {
                                                    let ax = (x as i128).abs();
                                                    max_abs = max_abs.max(ax);
                                                    if ax >= radix_half {
                                                        over_half += 1;
                                                    }
                                                    if ax >= radix_full {
                                                        over_full += 1;
                                                    }
                                                }

                                                global_max = global_max.max(max_abs);
                                                global_over_half += over_half;
                                                global_over_full += over_full;
                                                limb_max.push(max_abs);
                                                limb_over_half.push(over_half);
                                                limb_over_full.push(over_full);
                                            }

                                            println!(
                                                "[sheared phase5c2b width8-raw-limbs] backend={} label={} col={} k={} size={} limb_max={:?} over_2^{}={:?} over_2^{}={:?}",
                                                std::any::type_name::<BE>(),
                                                label,
                                                col,
                                                ct.k().as_usize(),
                                                ct.size(),
                                                limb_max,
                                                radix_bits - 1,
                                                limb_over_half,
                                                radix_bits,
                                                limb_over_full,
                                            );
                                        }

                                        let mut pair_sum_limb_max = Vec::with_capacity(ct.size());
                                        for limb in 0..ct.size() {
                                            let c0 = data.at(0, limb);
                                            let c1 = data.at(1, limb);
                                            let mut max_abs = 0i128;
                                            for (&x0, &x1) in c0.iter().zip(c1.iter()) {
                                                max_abs = max_abs.max(((x0 as i128) + (x1 as i128)).abs());
                                            }
                                            pair_sum_limb_max.push(max_abs);
                                        }

                                        let global_log2 = if global_max > 0 {
                                            (global_max as f64).log2()
                                        } else {
                                            f64::NEG_INFINITY
                                        };
                                        let pair_sum_global = pair_sum_limb_max.iter().copied().max().unwrap_or(0);
                                        let pair_sum_log2 = if pair_sum_global > 0 {
                                            (pair_sum_global as f64).log2()
                                        } else {
                                            f64::NEG_INFINITY
                                        };

                                        println!(
                                            "[sheared phase5c2b width8-raw-summary] backend={} label={} global_max={} global_log2={:.2} total_over_2^{}={} total_over_2^{}={} pair_sum_limb_max={:?} pair_sum_global={} pair_sum_log2={:.2}",
                                            std::any::type_name::<BE>(),
                                            label,
                                            global_max,
                                            global_log2,
                                            radix_bits - 1,
                                            global_over_half,
                                            radix_bits,
                                            global_over_full,
                                            pair_sum_limb_max,
                                            pair_sum_global,
                                            pair_sum_log2,
                                        );
                                    }
                                }

                                let (fresh_trunc_left_re, fresh_trunc_left_im) = ckks_decrypt_decode::<BE, F, E>(
                                    &phase5b_params,
                                    module,
                                    &encoder,
                                    &fresh_trunc_left,
                                    &phase5b_sk,
                                    &mut tree_scratch.borrow(),
                                );
                                let (fresh_trunc_right_re, fresh_trunc_right_im) = ckks_decrypt_decode::<BE, F, E>(
                                    &phase5b_params,
                                    module,
                                    &encoder,
                                    &fresh_trunc_right,
                                    &phase5b_sk,
                                    &mut tree_scratch.borrow(),
                                );

                                // ------------------------------------------------------------
                                // Decisive FFT64 tensor/relinearization split.
                                //
                                // Same k=171 historical/fresh operands as the
                                // truncation probe.  Here input k=171,
                                // log_delta=30, natural output k=141 and requested
                                // output k=141, so there is no extra destination
                                // narrowing.  The ct*ct convolution offset is 30.
                                //
                                // Decrypt the tensor BEFORE relinearization, then
                                // relinearize that exact tensor and decrypt again.
                                {
                                    use poulpy_hal::layouts::ZnxView as _;

                                    let probe_cnv_offset = trunc_delta;
                                    assert_eq!(trunc_k, 171);
                                    assert_eq!(trunc_left.k().as_usize(), trunc_k);
                                    assert_eq!(fresh_trunc_left.k().as_usize(), trunc_k);
                                    assert_eq!(fresh_trunc_right.k().as_usize(), trunc_k);
                                    assert_eq!(trunc_dst_k, trunc_budget);
                                    assert_eq!(probe_cnv_offset, 30);

                                    let tensor_layout = poulpy_core::layouts::GLWELayout {
                                        n: trunc_left.n(),
                                        base2k: trunc_left.base2k(),
                                        k: trunc_left.k(),
                                        rank: trunc_left.rank(),
                                    };
                                    let out_layout = poulpy_core::layouts::GLWELayout {
                                        n: trunc_left.n(),
                                        base2k: trunc_left.base2k(),
                                        k: trunc_dst_k.into(),
                                        rank: trunc_left.rank(),
                                    };

                                    let mut probe_sk_tensor = module.glwe_secret_tensor_alloc(trunc_left.rank());
                                    module.glwe_secret_tensor_prepare(&mut probe_sk_tensor, &tree_sk, &mut tree_scratch.borrow());
                                    let mut probe_sk_tensor_prepared =
                                        module.glwe_secret_tensor_prepared_alloc(trunc_left.rank());
                                    module.glwe_secret_tensor_prepared_prepare(&mut probe_sk_tensor_prepared, &probe_sk_tensor);

                                    let mut hist_tensor = module.glwe_tensor_alloc_from_infos(&tensor_layout);
                                    let mut fresh_tensor = module.glwe_tensor_alloc_from_infos(&tensor_layout);

                                    module.glwe_tensor_apply(
                                        probe_cnv_offset,
                                        &mut hist_tensor,
                                        &trunc_left,
                                        &fresh_trunc_right,
                                        &mut tree_scratch.borrow(),
                                    );
                                    module.glwe_tensor_apply(
                                        probe_cnv_offset,
                                        &mut fresh_tensor,
                                        &fresh_trunc_left,
                                        &fresh_trunc_right,
                                        &mut tree_scratch.borrow(),
                                    );

                                    let mut hist_tensor_pt = module.glwe_plaintext_alloc_from_infos(&tensor_layout);
                                    let mut fresh_tensor_pt = module.glwe_plaintext_alloc_from_infos(&tensor_layout);

                                    module.glwe_tensor_decrypt(
                                        &hist_tensor,
                                        &mut hist_tensor_pt,
                                        &phase5b_sk,
                                        &probe_sk_tensor_prepared,
                                        &mut tree_scratch.borrow(),
                                    );
                                    module.glwe_tensor_decrypt(
                                        &fresh_tensor,
                                        &mut fresh_tensor_pt,
                                        &phase5b_sk,
                                        &probe_sk_tensor_prepared,
                                        &mut tree_scratch.borrow(),
                                    );

                                    let mut hist_relin = module.glwe_alloc_from_infos(&out_layout);
                                    let mut fresh_relin = module.glwe_alloc_from_infos(&out_layout);
                                    module.glwe_tensor_relinearize(
                                        &mut hist_relin,
                                        &hist_tensor,
                                        &tensor_prepared,
                                        &mut tree_scratch.borrow(),
                                    );
                                    module.glwe_tensor_relinearize(
                                        &mut fresh_relin,
                                        &fresh_tensor,
                                        &tensor_prepared,
                                        &mut tree_scratch.borrow(),
                                    );

                                    let mut hist_relin_pt = module.glwe_plaintext_alloc_from_infos(&out_layout);
                                    let mut fresh_relin_pt = module.glwe_plaintext_alloc_from_infos(&out_layout);
                                    module.glwe_decrypt(&hist_relin, &mut hist_relin_pt, &phase5b_sk, &mut tree_scratch.borrow());
                                    module.glwe_decrypt(
                                        &fresh_relin,
                                        &mut fresh_relin_pt,
                                        &phase5b_sk,
                                        &mut tree_scratch.borrow(),
                                    );

                                    macro_rules! plaintext_limb_diff {
                                        ($lhs:expr, $rhs:expr) => {{
                                            let lhs_ref = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref($lhs);
                                            let rhs_ref = poulpy_core::layouts::GLWEToBackendRef::<BE>::to_backend_ref($rhs);
                                            let lhs_data = lhs_ref.data();
                                            let rhs_data = rhs_ref.data();
                                            let size = lhs_ref.size();
                                            assert_eq!(size, rhs_ref.size());

                                            let mut limb_max = Vec::with_capacity(size);
                                            let mut limb_nonzero = Vec::with_capacity(size);
                                            for limb in 0..size {
                                                let a = lhs_data.at(0, limb);
                                                let b = rhs_data.at(0, limb);
                                                let mut max_abs = 0i128;
                                                let mut nonzero = 0usize;
                                                for (&x, &y) in a.iter().zip(b.iter()) {
                                                    let d = (x as i128) - (y as i128);
                                                    max_abs = max_abs.max(d.abs());
                                                    if d != 0 {
                                                        nonzero += 1;
                                                    }
                                                }
                                                limb_max.push(max_abs);
                                                limb_nonzero.push(nonzero);
                                            }

                                            let highest_nonzero_limb = limb_max
                                                .iter()
                                                .rposition(|&x| x != 0)
                                                .map(|x| x as i64)
                                                .unwrap_or(-1);
                                            let global_max = limb_max.iter().copied().max().unwrap_or(0);
                                            let global_log2 = if global_max > 0 {
                                                (global_max as f64).log2()
                                            } else {
                                                f64::NEG_INFINITY
                                            };
                                            (
                                                limb_max,
                                                limb_nonzero,
                                                highest_nonzero_limb,
                                                global_max,
                                                global_log2,
                                            )
                                        }};
                                    }

                                    let (
                                        tensor_limb_max,
                                        tensor_limb_nonzero,
                                        tensor_highest,
                                        tensor_global_max,
                                        tensor_global_log2,
                                    ) = plaintext_limb_diff!(&hist_tensor_pt, &fresh_tensor_pt);

                                    let (relin_limb_max, relin_limb_nonzero, relin_highest, relin_global_max, relin_global_log2) =
                                        plaintext_limb_diff!(&hist_relin_pt, &fresh_relin_pt);

                                    println!(
                                        "[sheared phase5c2b width8-tensor-split] backend={} stage=tensor input_k={} tensor_k={} cnv_offset={} pt_size={} limb_max={:?} limb_nonzero={:?} highest_nonzero_limb={} global_max={} global_log2={:.2}",
                                        std::any::type_name::<BE>(),
                                        trunc_k,
                                        tensor_layout.k().as_usize(),
                                        probe_cnv_offset,
                                        hist_tensor_pt.size(),
                                        tensor_limb_max,
                                        tensor_limb_nonzero,
                                        tensor_highest,
                                        tensor_global_max,
                                        tensor_global_log2,
                                    );
                                    println!(
                                        "[sheared phase5c2b width8-tensor-split] backend={} stage=relin input_k={} out_k={} pt_size={} limb_max={:?} limb_nonzero={:?} highest_nonzero_limb={} global_max={} global_log2={:.2}",
                                        std::any::type_name::<BE>(),
                                        trunc_k,
                                        trunc_dst_k,
                                        hist_relin_pt.size(),
                                        relin_limb_max,
                                        relin_limb_nonzero,
                                        relin_highest,
                                        relin_global_max,
                                        relin_global_log2,
                                    );
                                }

                                let mut trunc_hist_fresh = alloc_ct(&phase5b_params, module, trunc_dst_k);
                                module
                                    .ckks_mul_into(
                                        &mut trunc_hist_fresh,
                                        &trunc_left,
                                        &fresh_trunc_right,
                                        &tensor_prepared,
                                        &mut tree_scratch.borrow(),
                                    )
                                    .unwrap();
                                let (trunc_hist_re, trunc_hist_im) = ckks_decrypt_decode::<BE, F, E>(
                                    &phase5b_params,
                                    module,
                                    &encoder,
                                    &trunc_hist_fresh,
                                    &phase5b_sk,
                                    &mut tree_scratch.borrow(),
                                );

                                let mut trunc_fresh_fresh = alloc_ct(&phase5b_params, module, trunc_dst_k);
                                module
                                    .ckks_mul_into(
                                        &mut trunc_fresh_fresh,
                                        &fresh_trunc_left,
                                        &fresh_trunc_right,
                                        &tensor_prepared,
                                        &mut tree_scratch.borrow(),
                                    )
                                    .unwrap();
                                let (trunc_fresh_re, trunc_fresh_im) = ckks_decrypt_decode::<BE, F, E>(
                                    &phase5b_params,
                                    module,
                                    &encoder,
                                    &trunc_fresh_fresh,
                                    &phase5b_sk,
                                    &mut tree_scratch.borrow(),
                                );

                                let mut trunc_hist_err = 0.0f64;
                                let mut trunc_fresh_err = 0.0f64;
                                let mut trunc_hist_max_abs = 0.0f64;
                                let mut trunc_fresh_max_abs = 0.0f64;
                                for p in 0..m {
                                    let lr = trunc_left_re[p].to_f64().unwrap();
                                    let li = trunc_left_im[p].to_f64().unwrap();
                                    let flr = fresh_trunc_left_re[p].to_f64().unwrap();
                                    let fli = fresh_trunc_left_im[p].to_f64().unwrap();
                                    let rr = fresh_trunc_right_re[p].to_f64().unwrap();
                                    let ri = fresh_trunc_right_im[p].to_f64().unwrap();
                                    let hist_pr = lr * rr - li * ri;
                                    let hist_pi = lr * ri + li * rr;
                                    let fresh_pr = flr * rr - fli * ri;
                                    let fresh_pi = flr * ri + fli * rr;

                                    let hr = trunc_hist_re[p].to_f64().unwrap();
                                    let hi = trunc_hist_im[p].to_f64().unwrap();
                                    let fr = trunc_fresh_re[p].to_f64().unwrap();
                                    let fi = trunc_fresh_im[p].to_f64().unwrap();

                                    trunc_hist_err = trunc_hist_err.max((hr - hist_pr).abs()).max((hi - hist_pi).abs());
                                    trunc_fresh_err = trunc_fresh_err.max((fr - fresh_pr).abs()).max((fi - fresh_pi).abs());
                                    trunc_hist_max_abs = trunc_hist_max_abs.max(hr.abs()).max(hi.abs());
                                    trunc_fresh_max_abs = trunc_fresh_max_abs.max(fr.abs()).max(fi.abs());
                                }

                                println!(
                                    "[sheared phase5c2b width8-truncate] backend={} trunc_k={} trunc_budget={} trunc_size={} trunc_max_k={} dst_k={} hist_x_fresh_err={:.3e} fresh_x_fresh_err={:.3e} max_abs_hist_x_fresh={:.3e} max_abs_fresh_x_fresh={:.3e}",
                                    std::any::type_name::<BE>(),
                                    trunc_k,
                                    trunc_budget,
                                    trunc_left.size(),
                                    trunc_left.max_k().as_usize(),
                                    trunc_hist_fresh.k().as_usize(),
                                    trunc_hist_err,
                                    trunc_fresh_err,
                                    trunc_hist_max_abs,
                                    trunc_fresh_max_abs,
                                );

                                // High-limb scaling probe.  Keep the exact same
                                // k=171 operands, but ask CKKS multiplication to
                                // round into progressively fewer output limbs.
                                // If a one-unit error in the highest retained radix
                                // limb is what becomes the ~1e30 decoded spike, each
                                // removed base-2^19 limb should lower log2(error) by
                                // approximately 19 bits.  Fresh x fresh is the control.
                                for scale_dst_k in [133usize, 114usize] {
                                    assert!(scale_dst_k > trunc_delta);

                                    let mut scale_hist = alloc_ct(&phase5b_params, module, scale_dst_k);
                                    module
                                        .ckks_mul_into(
                                            &mut scale_hist,
                                            &trunc_left,
                                            &fresh_trunc_right,
                                            &tensor_prepared,
                                            &mut tree_scratch.borrow(),
                                        )
                                        .unwrap();
                                    let (scale_hist_re, scale_hist_im) = ckks_decrypt_decode::<BE, F, E>(
                                        &phase5b_params,
                                        module,
                                        &encoder,
                                        &scale_hist,
                                        &phase5b_sk,
                                        &mut tree_scratch.borrow(),
                                    );

                                    let mut scale_fresh = alloc_ct(&phase5b_params, module, scale_dst_k);
                                    module
                                        .ckks_mul_into(
                                            &mut scale_fresh,
                                            &fresh_trunc_left,
                                            &fresh_trunc_right,
                                            &tensor_prepared,
                                            &mut tree_scratch.borrow(),
                                        )
                                        .unwrap();
                                    let (scale_fresh_re, scale_fresh_im) = ckks_decrypt_decode::<BE, F, E>(
                                        &phase5b_params,
                                        module,
                                        &encoder,
                                        &scale_fresh,
                                        &phase5b_sk,
                                        &mut tree_scratch.borrow(),
                                    );

                                    let mut scale_hist_err = 0.0f64;
                                    let mut scale_fresh_err = 0.0f64;
                                    let mut scale_hist_max_abs = 0.0f64;
                                    let mut scale_fresh_max_abs = 0.0f64;
                                    for p in 0..m {
                                        let lr = trunc_left_re[p].to_f64().unwrap();
                                        let li = trunc_left_im[p].to_f64().unwrap();
                                        let flr = fresh_trunc_left_re[p].to_f64().unwrap();
                                        let fli = fresh_trunc_left_im[p].to_f64().unwrap();
                                        let rr = fresh_trunc_right_re[p].to_f64().unwrap();
                                        let ri = fresh_trunc_right_im[p].to_f64().unwrap();
                                        let hist_pr = lr * rr - li * ri;
                                        let hist_pi = lr * ri + li * rr;
                                        let fresh_pr = flr * rr - fli * ri;
                                        let fresh_pi = flr * ri + fli * rr;

                                        let hr = scale_hist_re[p].to_f64().unwrap();
                                        let hi = scale_hist_im[p].to_f64().unwrap();
                                        let fr = scale_fresh_re[p].to_f64().unwrap();
                                        let fi = scale_fresh_im[p].to_f64().unwrap();

                                        scale_hist_err = scale_hist_err.max((hr - hist_pr).abs()).max((hi - hist_pi).abs());
                                        scale_fresh_err = scale_fresh_err.max((fr - fresh_pr).abs()).max((fi - fresh_pi).abs());
                                        scale_hist_max_abs = scale_hist_max_abs.max(hr.abs()).max(hi.abs());
                                        scale_fresh_max_abs = scale_fresh_max_abs.max(fr.abs()).max(fi.abs());
                                    }

                                    let dst_size = scale_hist.size();
                                    let top_limb_decode_bit = if dst_size == 0 {
                                        i64::MIN
                                    } else {
                                        ((dst_size - 1) * left.base2k().as_usize()) as i64 - trunc_delta as i64
                                    };
                                    let hist_log2 = if scale_hist_err > 0.0 {
                                        scale_hist_err.log2()
                                    } else {
                                        f64::NEG_INFINITY
                                    };
                                    let fresh_log2 = if scale_fresh_err > 0.0 {
                                        scale_fresh_err.log2()
                                    } else {
                                        f64::NEG_INFINITY
                                    };

                                    println!(
                                        "[sheared phase5c2b width8-limb-scale] backend={} input_k={} dst_k={} dst_size={} dst_max_k={} output_budget={} top_limb_decode_bit={} hist_x_fresh_err={:.3e} hist_log2_err={:.2} fresh_x_fresh_err={:.3e} fresh_log2_err={:.2} max_abs_hist={:.3e} max_abs_fresh={:.3e}",
                                        std::any::type_name::<BE>(),
                                        trunc_k,
                                        scale_hist.k().as_usize(),
                                        dst_size,
                                        scale_hist.max_k().as_usize(),
                                        scale_hist.log_budget(),
                                        top_limb_decode_bit,
                                        scale_hist_err,
                                        hist_log2,
                                        scale_fresh_err,
                                        fresh_log2,
                                        scale_hist_max_abs,
                                        scale_fresh_max_abs,
                                    );
                                }
                            }
                        }

                        println!(
                            "[sheared phase5c2b width8-isolate] backend={} input_k={} input_budget={} dst_k={} output_budget={} left_oracle_err={:.3e} right_oracle_err={:.3e} rotation_vs_decoded_err={:.3e} mul_vs_decoded_err={:.3e} control_mul_vs_decoded_err={:.3e} max_abs_left={:.3e} max_abs_right={:.3e} max_abs_rot={:.3e} max_abs_plain_product={:.3e} max_abs_out={:.3e} max_abs_control={:.3e}",
                            std::any::type_name::<BE>(),
                            left.k().as_usize(),
                            left.log_budget(),
                            out.k().as_usize(),
                            out.log_budget(),
                            left_oracle_err,
                            right_oracle_err,
                            rotation_vs_decoded_err,
                            mul_vs_decoded_err,
                            control_mul_vs_decoded_err,
                            max_abs_left,
                            max_abs_right,
                            max_abs_rot,
                            max_abs_decoded_product,
                            max_abs_out,
                            max_abs_control,
                        );
                    }

                    if pair_max_err >= PHASE5C2B_MAX_ABS_ERR {
                        println!(
                            "[sheared phase5c2b decode-diagnostic] backend={} width={} group={} max_err={:.3e} slot={}; continuing ciphertext arithmetic",
                            std::any::type_name::<BE>(),
                            out_width,
                            group,
                            pair_max_err,
                            pair_worst_slot,
                        );
                    }

                    next.push(out);
                }

                golden_cyclic = golden_cyclic_next!(golden_cyclic, width);
                let golden_tree_required_bits = if ld < 40 {
                    PHASE5C2A_MIN_BITS_FFT
                } else {
                    PHASE5C2A_MIN_BITS_EXACT
                };
                assert_packed_tree_matches_golden!(
                    format!("phase5c2b-width-{out_width}"),
                    &next,
                    out_width,
                    &golden_cyclic,
                    golden_tree_required_bits
                );

                let stage_output_budget = stage_output_budget.expect("Phase 5C-2b stage is non-empty");
                let stage_rms = (stage_sq_err / stage_components as f64).sqrt();
                let stage_worst_bits = if stage_worst_err == 0.0 {
                    f64::INFINITY
                } else {
                    -stage_worst_err.log2()
                };
                let stage_rms_bits = if stage_rms == 0.0 { f64::INFINITY } else { -stage_rms.log2() };

                let depth = out_width.ilog2() as usize;
                let expected_budget = initial_tree_budget
                    .checked_sub(depth * tree_log_delta)
                    .expect("Phase 5C-2b expected budget underflow");
                assert_eq!(
                    stage_output_budget, expected_budget,
                    "Phase 5C-2b budget trajectory differs from one log_delta consumed per product level"
                );

                println!(
                    "[sheared phase5c2b encrypted-tree-stage] backend={} width={} nodes={} api_rotation={} worst_group={} worst_slot={} worst_max_err={:.3e} ({:.1} bits) global_rms={:.3e} ({:.1} bits) input_log_budget={} output_log_budget={} expected_log_budget={}",
                    std::any::type_name::<BE>(),
                    out_width,
                    next.len(),
                    api_rotation,
                    stage_worst_group,
                    stage_worst_slot,
                    stage_worst_err,
                    stage_worst_bits,
                    stage_rms,
                    stage_rms_bits,
                    input_stage_budget,
                    stage_output_budget,
                    expected_budget,
                );

                tree_rotations += next.len();
                tree_multiplications += next.len();
                current = next;
                width = out_width;
            }

            assert_eq!(current.len(), 1);
            assert_eq!(width, groups);
            assert_eq!(tree_rotations, groups - 1);
            assert_eq!(tree_multiplications, groups - 1);
            assert_eq!(distinct_tree_rotation_keys, groups.ilog2() as usize);

            let root = &current[0];
            let final_expected_budget = initial_tree_budget
                .checked_sub((groups.ilog2() as usize) * tree_log_delta)
                .expect("Phase 5C-2b final expected budget underflow");
            assert_eq!(root.log_budget(), final_expected_budget);

            println!(
                "[sheared phase5c2b encrypted-tree] backend={} groups={} depth={} rotations={} multiplications={} distinct_rotation_keys={} api_rotations=[{}, {}, {}, {}, {}] input_log_budget={} root_log_budget={}",
                std::any::type_name::<BE>(),
                groups,
                groups.ilog2(),
                tree_rotations,
                tree_multiplications,
                distinct_tree_rotation_keys,
                m - 1,
                m - 2,
                m - 4,
                m - 8,
                m - 16,
                initial_tree_budget,
                root.log_budget(),
            );

            // -------------------------------------------------------------
            // Phase 5D: complete real Packed/Sheared-SHIP end-to-end.
            //
            // Match production real SHIP:
            //     output = root + Conj(root)
            // -------------------------------------------------------------
            let mut conjugated = alloc_ct(&phase5b_params, module, root.k().as_usize());
            module
                .ckks_conjugate_into(&mut conjugated, root, &ship_conj_prepared, &mut tree_scratch.borrow())
                .unwrap();

            let mut final_out = alloc_ct(&phase5b_params, module, root.k().as_usize());
            module
                .ckks_add_into(&mut final_out, root, &conjugated, &mut tree_scratch.borrow())
                .unwrap();
            final_out.set_slots(SlotsKind::Real);

            // GOLDEN-DIFF final gate: run the unmodified production SHIP API on
            // the same dense bottom ciphertext and the same prepared ShipKeySet.
            let mut golden_final = alloc_ct(&phase5b_params, module, kk);
            let golden_ship_bytes =
                CKKSShipOps::<BE, F>::ckks_ship_bootstrap_tmp_bytes(module, &golden_final, &ct0, &golden_keys).unwrap();
            let mut golden_ship_scratch = ScratchOwned::<BE>::alloc(golden_ship_bytes);
            CKKSShipOps::<BE, F>::ckks_ship_bootstrap_into(
                module,
                &mut golden_final,
                &ct0,
                &golden_keys,
                &mut golden_ship_scratch.borrow(),
            )
            .unwrap();

            let (final_re, final_im) = ckks_decrypt_decode::<BE, F, E>(
                &phase5b_params,
                module,
                &encoder,
                &final_out,
                &phase5b_sk,
                &mut tree_scratch.borrow(),
            );
            let (golden_re, golden_im) = ckks_decrypt_decode::<BE, F, E>(
                &phase5b_params,
                module,
                &encoder,
                &golden_final,
                &phase5b_sk,
                &mut golden_ship_scratch.borrow(),
            );

            let mut final_max_err = 0.0f64;
            let mut final_rms_acc = 0.0f64;
            let mut final_worst_slot = 0usize;
            for p in 0..m {
                let er = (final_re[p].to_f64().unwrap() - golden_re[p].to_f64().unwrap()).abs();
                let ei = (final_im[p].to_f64().unwrap() - golden_im[p].to_f64().unwrap()).abs();
                let local = er.max(ei);
                if local > final_max_err {
                    final_max_err = local;
                    final_worst_slot = p;
                }
                final_rms_acc += er * er + ei * ei;
            }

            let final_rms = (final_rms_acc / (2 * m) as f64).sqrt();
            let final_bits = if final_max_err == 0.0 {
                f64::INFINITY
            } else {
                -final_max_err.log2()
            };
            let final_required_bits = if ld < 40 { 7.0 } else { 11.0 };

            println!(
                "[golden-diff phase5d-full-ship] backend={} packed_root_k={} packed_root_budget={} packed_final_k={} packed_final_budget={} golden_final_k={} golden_final_budget={} max_err={:.3e} ({:.2} bits) rms={:.3e} worst_slot={}",
                std::any::type_name::<BE>(),
                root.k().as_usize(),
                root.log_budget(),
                final_out.k().as_usize(),
                final_out.log_budget(),
                golden_final.k().as_usize(),
                golden_final.log_budget(),
                final_max_err,
                final_bits,
                final_rms,
                final_worst_slot,
            );
            assert!(
                final_bits >= final_required_bits,
                "packed/sheared SHIP differs from production golden: {:.2} bits < {:.2} bits (max_err={:.3e}, slot={})",
                final_bits,
                final_required_bits,
                final_max_err,
                final_worst_slot,
            );
        }
    }

    let mut gen_key = |beta: bool, rot: usize, scratch: &mut ScratchOwned<BE>| {
        let key = hmux_rot_key_encrypt_sk(
            module,
            host_module,
            &sk_host,
            beta,
            rot,
            k,
            params.base2k.into(),
            mux_dsize,
            &mut xe,
            &mut xa,
            &mut scratch.borrow(),
        )
        .unwrap();
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key.key());
        module.glwe_switching_key_prepare(&mut prepared, key.key(), &mut scratch.borrow());
        HMuxRotKeyPrepared {
            key: prepared,
            gal_el: key.gal_el(),
        }
    };
    // Base-3 digit position of weight 5 with digit value 1 selected.
    let group_rot = vec![
        gen_key(false, 0, &mut scratch),
        gen_key(true, 5, &mut scratch),
        gen_key(false, 10, &mut scratch),
    ];
    let group_id = vec![gen_key(true, 0, &mut scratch), gen_key(false, 5, &mut scratch)];
    let group_zero = vec![gen_key(false, 0, &mut scratch), gen_key(false, 5, &mut scratch)];

    let mux_bytes = crate::default::ship::mux::ship_mux_rotate_tmp_bytes(
        module,
        &alloc_ct(&params, module, k),
        &group_rot[0].key,
        group_rot.len(),
    );
    let plans = crate::default::ship::mux::ship_mux_plans(
        module,
        [group_rot.as_slice(), group_id.as_slice(), group_zero.as_slice()].into_iter(),
    );
    let mut mux_scratch = ScratchOwned::<BE>::alloc(mux_bytes);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );

    // digit = 1, weight = 5: paper Rot_5 = poulpy rotation by -5.
    let (want_re, want_im) = want_rotate(&re1, &im1, -5, m);
    let mut out = alloc_ct(&params, module, k);
    module.ckks_copy(&mut out, &ct, &mut scratch.borrow()).unwrap();
    crate::default::ship::mux::ship_mux_rotate(module, &mut out, &group_rot, &plans, &mut mux_scratch.borrow()).unwrap();
    assert_eq!(out.log_delta(), ct.log_delta());
    assert_eq!(out.log_budget(), ct.log_budget());
    assert_decrypt_precision(
        "mux_rotate(digit=1,weight=5)",
        &params,
        module,
        &encoder,
        &out,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    // digit = 0: identity.
    let mut out_id = alloc_ct(&params, module, k);
    module.ckks_copy(&mut out_id, &ct, &mut scratch.borrow()).unwrap();
    crate::default::ship::mux::ship_mux_rotate(module, &mut out_id, &group_id, &plans, &mut mux_scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "mux_rotate(digit=0)",
        &params,
        module,
        &encoder,
        &out_id,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );

    // No digit selected: all-zero cleartext.
    let zeros = vec![F::zero(); m];
    let mut out_zero = alloc_ct(&params, module, k);
    module.ckks_copy(&mut out_zero, &ct, &mut scratch.borrow()).unwrap();
    crate::default::ship::mux::ship_mux_rotate(module, &mut out_zero, &group_zero, &plans, &mut mux_scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "mux_rotate(empty)",
        &params,
        module,
        &encoder,
        &out_zero,
        &sk,
        &zeros,
        &zeros,
        &mut scratch.borrow(),
    );
}

/// End-to-end SHIP bootstrap driver: a one-limb bottom ciphertext under the
/// dense key, holding cleartexts with gap `gamma` in its coefficients (real:
/// first `N/2` only; complex: `Re`/`Im` halves), is lifted to a slots-domain
/// ciphertext at the raised modulus.
fn ship_bootstrap_case<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    complex: bool,
) where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: TestContextModule<BE>
        + CKKSShipOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let n = params.n;
    let m = n / 2;
    let base2k = params.base2k;
    let encoder = super::reference_encoder::ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let plan = ship_suite_plan(&params);
    let kk = plan.raised_k(base2k);
    let big = CKKSTestParams { k: kk, ..params };

    let (sk_dense_host, _, sk_dense) = gen_sk_with_host(&big, module, host_module, [0u8; 32]);
    let mut source = Source::new([9u8; 32]);
    let spec = ShipSecretSpec::sample(&plan, &mut source);

    let keys_layout = ShipKeysLayout {
        mux_dsize: params.dsize,
        tensor_key: big.tsk_layout().layout,
        conjugation_key: big.atk_layout().layout,
        complex,
    };
    let mut scratch = alloc_scratch(&big, module);
    let mut xe = Source::new([11u8; 32]);
    let mut xa = Source::new([12u8; 32]);
    let key_set = ShipKeySet::generate::<BE, F>(
        module,
        host_module,
        &plan,
        base2k.into(),
        &spec,
        &sk_dense_host,
        &keys_layout,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    )
    .unwrap();
    let keys = key_set.prepare(module, &mut scratch.borrow()).unwrap();

    // Bottom ciphertext: torus content sum round(q0 * mu_i / gamma) X^i, i.e.
    // value mu_i / gamma at log_delta = base2k with no budget; the complex
    // case also carries Im(mu_i) in coefficient i + m.
    let gamma = 2f64.powi(plan.log_gamma() as i32);
    let mu_re: Vec<f64> = (0..m).map(|i| ((i as f64 * 0.7639) % 1.0 - 0.5) * 0.9).collect();
    let mu_im: Vec<f64> = (0..m)
        .map(|i| {
            if complex {
                ((i as f64 * 0.4117 + 0.31) % 1.0 - 0.5) * 0.9
            } else {
                0.0
            }
        })
        .collect();
    let coeffs: Vec<F> = (0..n)
        .map(|i| {
            if i < m {
                F::from_f64(mu_re[i] / gamma).unwrap()
            } else {
                F::from_f64(mu_im[i - m] / gamma).unwrap()
            }
        })
        .collect();
    let bottom_prec = ckks_spec(n, base2k, base2k, 0);
    let mut host_pt = host_module.ckks_pt_vec_alloc(base2k.into(), bottom_prec.k());
    host_pt.set_meta(bottom_prec.meta());
    host_pt.encode_host_floats(&coeffs).unwrap();
    let mut coeffs_quant = vec![F::zero(); n];
    host_pt.decode_host_floats(&mut coeffs_quant).unwrap();
    let mu_quant_re: Vec<f64> = (0..m).map(|i| gamma * coeffs_quant[i].to_f64().unwrap()).collect();
    let mu_quant_im: Vec<f64> = (0..m).map(|i| gamma * coeffs_quant[i + m].to_f64().unwrap()).collect();
    let ct0 = ckks_encrypt_pt(&big, module, &sk_dense, base2k, &host_pt, &mut scratch.borrow());

    let mut out = alloc_ct(&big, module, kk);
    let ship_bytes = CKKSShipOps::<BE, F>::ckks_ship_bootstrap_tmp_bytes(module, &out, &ct0, &keys).unwrap();
    let mut ship_scratch = ScratchOwned::<BE>::alloc(ship_bytes);
    if complex {
        CKKSShipOps::<BE, F>::ckks_ship_bootstrap_complex_into(module, &mut out, &ct0, &keys, &mut ship_scratch.borrow())
            .unwrap();
    } else {
        CKKSShipOps::<BE, F>::ckks_ship_bootstrap_into(module, &mut out, &ct0, &keys, &mut ship_scratch.borrow()).unwrap();
    }
    assert_eq!(
        out.slots(),
        if complex { SlotsKind::Complex } else { SlotsKind::Real },
        "ship_bootstrap: output slot kind"
    );
    assert!(
        out.log_budget() >= plan.log_budget_out(),
        "ship_bootstrap: no budget regained (log_budget={} < {})",
        out.log_budget(),
        plan.log_budget_out()
    );

    let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(&big, module, &encoder, &out, &sk_dense, &mut scratch.borrow());
    let mut max_err: f64 = 0.0;
    for i in 0..m {
        let er = (got_re[i].to_f64().unwrap() - mu_quant_re[i]).abs();
        let ei = (got_im[i].to_f64().unwrap() - mu_quant_im[i]).abs();
        max_err = max_err.max(er).max(ei);
    }
    let measured_bits = -max_err.log2();
    // Gap model: error ~ (2*pi)^2 * mu^3 / (6 * gamma^2), 12.74 bits for
    // gamma = 2^6 and |mu| <= 0.45; exact backends measure exactly that.
    // FFT64 (f64 arithmetic) loses ~3 bits over the deep keyswitch chain and
    // ~2 more to the encapsulation switch at its toy base2k bottom modulus.
    let required_bits = if plan.log_delta_work() < 40 { 7.0 } else { 12.0 };

    // Optional human-readable I/O dump for inspecting one real end-to-end SHIP
    // test case.  Keep normal test output quiet unless explicitly requested.
    if std::env::var_os("POULPY_SHIP_SHOW_IO").is_some() {
        let show = m.min(8);
        println!("\n=== SHIP bootstrap I/O ({}) ===", if complex { "complex" } else { "real" });
        println!(
            "N={n}, slots={m}, gamma=2^{}={}, raised_k={kk}, log_budget_out={}, max_err={max_err:.6e}, measured_bits={measured_bits:.3}, required_bits={required_bits:.3}",
            plan.log_gamma(),
            gamma,
            out.log_budget(),
        );
        println!(
            "{:<5} | {:>12} {:>12} | {:>12} {:>12} | {:>12} {:>12} | {:>10}",
            "slot", "input_re", "input_im", "expect_re", "expect_im", "got_re", "got_im", "max_abs_err"
        );
        println!("{}", "-".repeat(105));
        for i in 0..show {
            let gr = got_re[i].to_f64().unwrap();
            let gi = got_im[i].to_f64().unwrap();
            let er = (gr - mu_quant_re[i]).abs();
            let ei = (gi - mu_quant_im[i]).abs();
            println!(
                "{:<5} | {:>12.8} {:>12.8} | {:>12.8} {:>12.8} | {:>12.8} {:>12.8} | {:>10.3e}",
                i,
                mu_re[i],
                mu_im[i],
                mu_quant_re[i],
                mu_quant_im[i],
                gr,
                gi,
                er.max(ei),
            );
        }
        println!("input_* = requested cleartext; expect_* = bottom-encoding-quantized cleartext actually used by the assertion");
        println!("================================\n");
    }

    assert!(
        measured_bits >= required_bits,
        "ship_bootstrap: precision {measured_bits:.2} bits < required {required_bits:.2}"
    );
}

/// End-to-end SHIP half bootstrap over real cleartexts.
pub fn test_ship_bootstrap<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: TestContextModule<BE>
        + CKKSShipOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    ship_bootstrap_case::<BE, F, E>(params, module, host_module, false);
}

/// End-to-end complex SHIP bootstrap: `Re`/`Im` coefficient halves refreshed
/// into complex slots through one shared selector-mask family, the fixed
/// omega_2 pi permutation, and shared H-MUX/tensor-key dataflow.
pub fn test_ship_bootstrap_complex<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    Module<BE>: TestContextModule<BE>
        + CKKSShipOps<BE, F>
        + CKKSEncodingOps<BE, F>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvPVecBytesOf,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + ShipScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    ship_bootstrap_case::<BE, F, E>(params, module, host_module, true);
}
