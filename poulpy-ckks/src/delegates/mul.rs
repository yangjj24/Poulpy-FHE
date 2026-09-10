use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, VecZnxCopyBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::api::CKKSMulOps;

use crate::{CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos, layouts::CKKSPreparedRight, oep::CKKSMulImpl};

/// The precision each ciphertext-ciphertext operation resolves its key at.
///
/// One definition per operation, used by the scratch query and by the execution
/// alike, so the two cannot drift apart. The assign forms read their
/// destination as an operand, which is why it enters here and not only there.
fn mul_k<A: CKKSCtBounds, B: CKKSCtBounds>(a: &A, b: &B) -> TorusPrecision {
    a.k().max(b.k())
}

fn square_k<A: CKKSCtBounds>(a: &A) -> TorusPrecision {
    a.k()
}

fn prepared_mul_k_checked<D: CKKSCtBounds>(dst: &D, prepared_k: usize) -> Result<TorusPrecision> {
    let prepared_k = u32::try_from(prepared_k)
        .map_err(|_| crate::CKKSError::Internal(anyhow::anyhow!("prepared precision {prepared_k} exceeds u32")))?;
    Ok(dst.k().max(TorusPrecision(prepared_k)))
}

impl<BE: Backend + CKKSMulImpl<BE>> CKKSMulOps<BE> for Module<BE>
where
    Module<BE>: GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulConst<BE>
        + GLWEMulPlain<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxCopyBackend<BE>,
{
    fn ckks_mul_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // Sizing takes the key, not a decomposition: which effective `dsize` the
        // helper dispatches at is resolved during execution, and the budget
        // below covers every one this key admits.
        BE::ckks_mul_tmp_bytes_impl(self, res, a, b, tsk)
    }

    fn ckks_mul_dual_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_mul_dual_tmp_bytes_impl(self, res, a, b, tsk)
    }

    fn ckks_square_tmp_bytes<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_square_tmp_bytes_impl(self, res, a, tsk)
    }

    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_pt_vec_tmp_bytes_impl(self, res, a, b.k())
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_pt_const_tmp_bytes_impl(self, res, a, b.k())
    }

    fn ckks_mul_into<Dst, A, B, H>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        let k = mul_k(a, b);
        tsk.get_tensor_key(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_into",
                k: k.into(),
            })?;
        BE::ckks_mul_into_impl(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_into_dual<Dst, A, B, H>(
        &self,
        dst0: &mut Dst,
        a0: &A,
        b0: &B,
        dst1: &mut Dst,
        a1: &A,
        b1: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        let k0 = mul_k(a0, b0);
        let k1 = mul_k(a1, b1);
        ckks_ensure!(k0 == k1, "ckks_mul_into_dual: tensor precisions differ");
        ckks_ensure!(
            dst0.n() == dst1.n() && dst0.base2k() == dst1.base2k() && dst0.rank() == dst1.rank() && dst0.k() == dst1.k(),
            "ckks_mul_into_dual: destination layouts differ"
        );
        ckks_ensure!(
            a0.n() == a1.n()
                && a0.base2k() == a1.base2k()
                && a0.rank() == a1.rank()
                && a0.k() == a1.k()
                && b0.n() == b1.n()
                && b0.base2k() == b1.base2k()
                && b0.rank() == b1.rank()
                && b0.k() == b1.k(),
            "ckks_mul_into_dual: operand layouts differ"
        );
        tsk.get_tensor_key(k0)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_into_dual",
                k: k0.into(),
            })?;
        BE::ckks_mul_into_dual_impl(self, dst0, a0, b0, dst1, a1, b1, tsk, scratch)
    }

    fn ckks_mul_assign<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        let k = mul_k(dst, a);
        tsk.get_tensor_key(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_assign",
                k: k.into(),
            })?;
        BE::ckks_mul_assign_impl(self, dst, a, tsk, scratch)
    }

    fn ckks_prepare_right<A>(&self, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_right_impl(self, a, scratch)
    }

    fn ckks_mul_prepared_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        let k = prepared_mul_k_checked(dst, prepared.k)?;
        tsk.get_tensor_key(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_prepared_assign",
                k: k.into(),
            })?;
        BE::ckks_mul_prepared_assign_impl(self, dst, prepared, tsk, scratch)
    }

    fn ckks_square_into<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        let k = square_k(a);
        tsk.get_tensor_key(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_into",
                k: k.into(),
            })?;
        BE::ckks_square_into_impl(self, dst, a, tsk, scratch)
    }

    fn ckks_square_assign<Dst, H>(&self, dst: &mut Dst, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE>,
    {
        let k = square_k(dst);
        tsk.get_tensor_key(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_assign",
                k: k.into(),
            })?;
        BE::ckks_square_assign_impl(self, dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_into_impl(self, dst, a, pt, scratch)
    }

    fn ckks_mul_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_assign_impl(self, dst, pt, scratch)
    }

    fn ckks_mul_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_into_impl(self, dst, a, pt, pt_coeff, scratch)
    }

    fn ckks_mul_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_assign_impl(self, dst, pt, pt_coeff, scratch)
    }
}
