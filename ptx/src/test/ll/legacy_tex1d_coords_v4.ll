%struct.texture = type { %struct.textureReference }
%struct.textureReference = type { i32, i32, i32, [3 x i32], %struct.hipChannelFormatDesc, i32, i32, i32, float, float, float, ptr, i32, i32 }
%struct.hipChannelFormatDesc = type { i32, i32, i32, i32, i32 }

@sample_tex = addrspace(1) externally_initialized global %struct.texture undef, align 8

; Function Attrs: denormal_fpenv(dynamic)
declare hidden <4 x float> @__zluda_ptx_impl_texref_1d_v4_f32_s32(ptr addrspace(1), <1 x i32>) #0

; Function Attrs: denormal_fpenv(preservesign)
define amdgpu_kernel void @fetch_1d(ptr addrspace(4) byref(i64) %"52") #1 {
  %"53" = alloca i64, align 8, addrspace(5)
  %"54" = alloca i32, align 4, addrspace(5)
  %"55" = alloca i32, align 4, addrspace(5)
  %"56" = alloca i32, align 4, addrspace(5)
  %"57" = alloca i32, align 4, addrspace(5)
  %"58" = alloca float, align 4, addrspace(5)
  %"59" = alloca float, align 4, addrspace(5)
  %"60" = alloca float, align 4, addrspace(5)
  %"61" = alloca float, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"51"

"51":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"52", align 8
  store i64 %2, ptr addrspace(5) %"53", align 8
  store i32 1, ptr addrspace(5) %"54", align 4
  store i32 17, ptr addrspace(5) %"55", align 4
  store i32 -1, ptr addrspace(5) %"56", align 4
  store i32 91, ptr addrspace(5) %"57", align 4
  %3 = load i32, ptr addrspace(5) %"54", align 4
  %"49" = insertelement <1 x i32> undef, i32 %3, i8 0
  %"48" = call <4 x float> @__zluda_ptx_impl_texref_1d_v4_f32_s32(ptr addrspace(1) @sample_tex, <1 x i32> %"49")
  %"68" = extractelement <4 x float> %"48", i8 0
  %"69" = extractelement <4 x float> %"48", i8 1
  %"70" = extractelement <4 x float> %"48", i8 2
  %"71" = extractelement <4 x float> %"48", i8 3
  store float %"68", ptr addrspace(5) %"58", align 4
  store float %"69", ptr addrspace(5) %"59", align 4
  store float %"70", ptr addrspace(5) %"60", align 4
  store float %"71", ptr addrspace(5) %"61", align 4
  %4 = load float, ptr addrspace(5) %"58", align 4
  %5 = load float, ptr addrspace(5) %"59", align 4
  %6 = load float, ptr addrspace(5) %"60", align 4
  %7 = load float, ptr addrspace(5) %"61", align 4
  %8 = insertelement <4 x float> undef, float %4, i8 0
  %9 = insertelement <4 x float> %8, float %5, i8 1
  %10 = insertelement <4 x float> %9, float %6, i8 2
  %"50" = insertelement <4 x float> %10, float %7, i8 3
  %11 = load i64, ptr addrspace(5) %"53", align 8
  %"78" = inttoptr i64 %11 to ptr addrspace(1)
  store <4 x float> %"50", ptr addrspace(1) %"78", align 16
  ret void
}

attributes #0 = { denormal_fpenv(dynamic) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { denormal_fpenv(preservesign) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
