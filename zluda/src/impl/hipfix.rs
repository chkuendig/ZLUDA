// There's a bug in hipDrvPointerGetAttributes where it returns
// HIP_ERROR_INVALID_VALUE if the pointer is null. It works correctly for any
// other invalid pointer
pub(crate) fn get_attributes(
    ptr: hip_runtime_sys::hipDeviceptr_t,
) -> hip_runtime_sys::hipDeviceptr_t {
    if ptr.0.is_null() {
        hip_runtime_sys::hipDeviceptr_t(usize::MAX as _)
    } else {
        ptr
    }
}

// For some reason HIP does not tolerate hipArraySurfaceLoadStore, even though
// it works just fine
pub(crate) unsafe fn array_3d_create(descriptor: &mut hip_runtime_sys::HIP_ARRAY3D_DESCRIPTOR) {
    descriptor.Flags &= !hip_runtime_sys::hipArraySurfaceLoadStore;
}

#[must_use]
pub(crate) fn get_non_broken_format(
    format: hip_runtime_sys::hipArray_Format,
) -> hip_runtime_sys::hipArray_Format {
    use hip_runtime_sys::hipArray_Format;
    // These formats cause issues in HIP — workaround by using compatible unsigned types
    match format {
        hipArray_Format::HIP_AD_FORMAT_HALF => hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16,
        hipArray_Format::HIP_AD_FORMAT_SIGNED_INT16 => {
            hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT16
        }
        hipArray_Format::HIP_AD_FORMAT_SIGNED_INT8 => {
            hipArray_Format::HIP_AD_FORMAT_UNSIGNED_INT8
        }
        f => f,
    }
}
