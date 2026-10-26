; Active shuffle sources must retain their exact values. strict.wwm may not
; propagate into their guarded producers. set.inactive is the AMDGPU entry
; boundary for this; poison leaves inactive values unspecified rather than
; forcing them to zero. Only ds_bpermute is wrapped in strict.wwm by the caller.
;
; This is IR because a C++ declaration of set.inactive adds noundef to its
; arguments, making an explicit poison argument invalid. Keep this shim free
; of noundef on the inactive operand and link it before optimising the helper.
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32) convergent

define linkonce_odr i32 @__zluda_ptx_impl_shfl_snapshot_i32(i32 %input) alwaysinline convergent nounwind {
    %snapshot = call i32 @llvm.amdgcn.set.inactive.i32(i32 %input, i32 poison)
    ret i32 %snapshot
}
