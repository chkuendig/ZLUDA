target triple = "amdgcn-amd-amdhsa"
declare <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32.v8i32.v4i32(i32 immarg, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) nounwind
declare float @llvm.sin.f32(float) nounwind

define <4 x float> @__llvm_amdgcn_image_transparent(float %x, float %y, float %z, <8 x i32> %image, <4 x i32> %sampler) {
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32.v8i32.v4i32(i32 15, float %x, float %y, float %z, <8 x i32> %image, <4 x i32> %sampler, i1 false, i32 0, i32 0)
  ret <4 x float> %v
}
define <4 x float> @strict_caller(float %x, <8 x i32> %image, <4 x i32> %sampler) strictfp "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" {
  %v = call <4 x float> @__llvm_amdgcn_image_transparent(float %x, float 1.0, float 2.0, <8 x i32> %image, <4 x i32> %sampler)
  ret <4 x float> %v
}
define <4 x float> @ieee_caller(float %x, <8 x i32> %image, <4 x i32> %sampler) "denormal-fp-math"="ieee" "denormal-fp-math-f32"="ieee" {
  %v = call <4 x float> @__llvm_amdgcn_image_transparent(float %x, float 1.0, float 2.0, <8 x i32> %image, <4 x i32> %sampler)
  ret <4 x float> %v
}
define <4 x float> @__llvm_amdgcn_image_arithmetic(float %x, <8 x i32> %image, <4 x i32> %sampler) {
  %adjusted = fadd float %x, 1.0
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32.v8i32.v4i32(i32 15, float %adjusted, float 1.0, float 2.0, <8 x i32> %image, <4 x i32> %sampler, i1 false, i32 0, i32 0)
  ret <4 x float> %v
}
define <4 x float> @arithmetic_caller(float %x, <8 x i32> %image, <4 x i32> %sampler) {
  %v = call <4 x float> @__llvm_amdgcn_image_arithmetic(float %x, <8 x i32> %image, <4 x i32> %sampler)
  ret <4 x float> %v
}
define <4 x float> @__llvm_amdgcn_image_branch(float %x, <8 x i32> %image, <4 x i32> %sampler) {
entry:
  br label %sample
sample:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.3d.v4f32.f32.v8i32.v4i32(i32 15, float %x, float 1.0, float 2.0, <8 x i32> %image, <4 x i32> %sampler, i1 false, i32 0, i32 0)
  ret <4 x float> %v
}
define <4 x float> @branch_caller(float %x, <8 x i32> %image, <4 x i32> %sampler) {
  %v = call <4 x float> @__llvm_amdgcn_image_branch(float %x, <8 x i32> %image, <4 x i32> %sampler)
  ret <4 x float> %v
}
define float @__llvm_amdgcn_image_not_an_image(float %x) {
  %v = call float @llvm.sin.f32(float %x)
  ret float %v
}
define float @non_image_caller(float %x) {
  %v = call float @__llvm_amdgcn_image_not_an_image(float %x)
  ret float %v
}
define <4 x float> @noinline_caller(float %x, <8 x i32> %image, <4 x i32> %sampler) {
  %v = call <4 x float> @__llvm_amdgcn_image_transparent(float %x, float 1.0, float 2.0, <8 x i32> %image, <4 x i32> %sampler) noinline
  ret <4 x float> %v
}
