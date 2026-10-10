declare hidden <4 x float> @__zluda_ptx_impl_tex_2d_v4_f32_f32_obj(i64, <2 x float>) #0

define amdgpu_kernel void @tex_obj_stub(ptr addrspace(4) byref(i64) %"48") #1 {
  %"49" = alloca i64, align 8, addrspace(5)
  %"50" = alloca i64, align 8, addrspace(5)
  %"51" = alloca float, align 4, addrspace(5)
  %"52" = alloca float, align 4, addrspace(5)
  %"53" = alloca float, align 4, addrspace(5)
  %"54" = alloca float, align 4, addrspace(5)
  %"55" = alloca float, align 4, addrspace(5)
  %"56" = alloca float, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"47"

"47":                                             ; preds = %1
  fence seq_cst
  %2 = load i64, ptr addrspace(4) %"48", align 8
  store i64 %2, ptr addrspace(5) %"49", align 8
  store i64 0, ptr addrspace(5) %"50", align 8
  store float 0.000000e+00, ptr addrspace(5) %"51", align 4
  store float 0.000000e+00, ptr addrspace(5) %"52", align 4
  %3 = load float, ptr addrspace(5) %"51", align 4
  %4 = load float, ptr addrspace(5) %"52", align 4
  %5 = insertelement <2 x float> zeroinitializer, float %3, i8 0
  %"46" = insertelement <2 x float> %5, float %4, i8 1
  %6 = load i64, ptr addrspace(5) %"50", align 8
  %"45" = call <4 x float> @__zluda_ptx_impl_tex_2d_v4_f32_f32_obj(i64 %6, <2 x float> %"46")
  %"64" = extractelement <4 x float> %"45", i8 0
  %"65" = extractelement <4 x float> %"45", i8 1
  %"66" = extractelement <4 x float> %"45", i8 2
  %"67" = extractelement <4 x float> %"45", i8 3
  store float %"64", ptr addrspace(5) %"53", align 4
  store float %"65", ptr addrspace(5) %"54", align 4
  store float %"66", ptr addrspace(5) %"55", align 4
  store float %"67", ptr addrspace(5) %"56", align 4
  %7 = load i64, ptr addrspace(5) %"49", align 8
  %8 = load float, ptr addrspace(5) %"53", align 4
  %"70" = inttoptr i64 %7 to ptr addrspace(1)
  store float %8, ptr addrspace(1) %"70", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="dynamic" "denormal-fp-math-f32"="dynamic" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
