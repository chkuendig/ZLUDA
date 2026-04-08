define amdgpu_kernel void @histogram_word1(ptr addrspace(4) byref(i64) %"86", ptr addrspace(4) byref(i64) %"87") #0 {
  %"88" = alloca i64, align 8, addrspace(5)
  %"89" = alloca i64, align 8, addrspace(5)
  %"90" = alloca i32, align 4, addrspace(5)
  %"91" = alloca i32, align 4, addrspace(5)
  %"92" = alloca i32, align 4, addrspace(5)
  %"93" = alloca i32, align 4, addrspace(5)
  %"94" = alloca i32, align 4, addrspace(5)
  %"95" = alloca i32, align 4, addrspace(5)
  %"96" = alloca i32, align 4, addrspace(5)
  %"97" = alloca i32, align 4, addrspace(5)
  %"98" = alloca i32, align 4, addrspace(5)
  %"99" = alloca i32, align 4, addrspace(5)
  %"100" = alloca i32, align 4, addrspace(5)
  %"101" = alloca i32, align 4, addrspace(5)
  %"102" = alloca i32, align 4, addrspace(5)
  %"103" = alloca i32, align 4, addrspace(5)
  %"104" = alloca i32, align 4, addrspace(5)
  %"105" = alloca i32, align 4, addrspace(5)
  %"106" = alloca i32, align 4, addrspace(5)
  %"107" = alloca i32, align 4, addrspace(5)
  %"108" = alloca i32, align 4, addrspace(5)
  %"109" = alloca i32, align 4, addrspace(5)
  %"110" = alloca i32, align 4, addrspace(5)
  %"111" = alloca i32, align 4, addrspace(5)
  %"112" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"85"

"85":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"86", align 8
  store i64 %2, ptr addrspace(5) %"88", align 8
  %3 = load i64, ptr addrspace(4) %"87", align 8
  store i64 %3, ptr addrspace(5) %"89", align 8
  %4 = load i64, ptr addrspace(5) %"88", align 8
  %"196" = inttoptr i64 %4 to ptr
  %5 = load i32, ptr %"196", align 4
  store i32 %5, ptr addrspace(5) %"90", align 4
  %6 = load i64, ptr addrspace(5) %"88", align 8
  %"197" = inttoptr i64 %6 to ptr
  %"61" = getelementptr inbounds i8, ptr %"197", i64 4
  %7 = load i32, ptr %"61", align 4
  store i32 %7, ptr addrspace(5) %"91", align 4
  %8 = load i64, ptr addrspace(5) %"88", align 8
  %"198" = inttoptr i64 %8 to ptr
  %"63" = getelementptr inbounds i8, ptr %"198", i64 8
  %9 = load i32, ptr %"63", align 4
  store i32 %9, ptr addrspace(5) %"92", align 4
  %10 = load i64, ptr addrspace(5) %"88", align 8
  %"199" = inttoptr i64 %10 to ptr
  %"65" = getelementptr inbounds i8, ptr %"199", i64 12
  %11 = load i32, ptr %"65", align 4
  store i32 %11, ptr addrspace(5) %"93", align 4
  store i32 0, ptr addrspace(5) %"112", align 4
  %12 = load i32, ptr addrspace(5) %"90", align 4
  %13 = load i32, ptr addrspace(5) %"112", align 4
  %14 = icmp uge i32 %13, 32
  %15 = lshr i32 %12, %13
  %"124" = select i1 %14, i32 0, i32 %15
  store i32 %"124", ptr addrspace(5) %"94", align 4
  %16 = load i32, ptr addrspace(5) %"91", align 4
  %17 = load i32, ptr addrspace(5) %"112", align 4
  %18 = icmp uge i32 %17, 32
  %19 = lshr i32 %16, %17
  %"127" = select i1 %18, i32 0, i32 %19
  store i32 %"127", ptr addrspace(5) %"95", align 4
  %20 = load i32, ptr addrspace(5) %"92", align 4
  %21 = load i32, ptr addrspace(5) %"112", align 4
  %22 = icmp uge i32 %21, 32
  %23 = lshr i32 %20, %21
  %"130" = select i1 %22, i32 0, i32 %23
  store i32 %"130", ptr addrspace(5) %"96", align 4
  %24 = load i32, ptr addrspace(5) %"93", align 4
  %25 = load i32, ptr addrspace(5) %"112", align 4
  %26 = icmp uge i32 %25, 32
  %27 = lshr i32 %24, %25
  %"133" = select i1 %26, i32 0, i32 %27
  store i32 %"133", ptr addrspace(5) %"97", align 4
  %28 = load i32, ptr addrspace(5) %"94", align 4
  %"200" = and i32 %28, 15
  store i32 %"200", ptr addrspace(5) %"94", align 4
  %29 = load i32, ptr addrspace(5) %"95", align 4
  %"202" = and i32 %29, 15
  store i32 %"202", ptr addrspace(5) %"95", align 4
  %30 = load i32, ptr addrspace(5) %"96", align 4
  %"204" = and i32 %30, 15
  store i32 %"204", ptr addrspace(5) %"96", align 4
  %31 = load i32, ptr addrspace(5) %"97", align 4
  %"206" = and i32 %31, 15
  store i32 %"206", ptr addrspace(5) %"97", align 4
  %32 = load i32, ptr addrspace(5) %"94", align 4
  %33 = shl i32 %32, 4
  %"208" = select i1 false, i32 0, i32 %33
  store i32 %"208", ptr addrspace(5) %"98", align 4
  %34 = load i32, ptr addrspace(5) %"95", align 4
  %35 = shl i32 %34, 4
  %"210" = select i1 false, i32 0, i32 %35
  store i32 %"210", ptr addrspace(5) %"99", align 4
  %36 = load i32, ptr addrspace(5) %"96", align 4
  %37 = shl i32 %36, 4
  %"212" = select i1 false, i32 0, i32 %37
  store i32 %"212", ptr addrspace(5) %"100", align 4
  %38 = load i32, ptr addrspace(5) %"97", align 4
  %39 = shl i32 %38, 4
  %"214" = select i1 false, i32 0, i32 %39
  store i32 %"214", ptr addrspace(5) %"101", align 4
  %40 = load i32, ptr addrspace(5) %"98", align 4
  %"216" = and i32 %40, 240
  store i32 %"216", ptr addrspace(5) %"98", align 4
  %41 = load i32, ptr addrspace(5) %"99", align 4
  %"218" = and i32 %41, 240
  store i32 %"218", ptr addrspace(5) %"99", align 4
  %42 = load i32, ptr addrspace(5) %"100", align 4
  %"220" = and i32 %42, 240
  store i32 %"220", ptr addrspace(5) %"100", align 4
  %43 = load i32, ptr addrspace(5) %"101", align 4
  %"222" = and i32 %43, 240
  store i32 %"222", ptr addrspace(5) %"101", align 4
  store i32 1, ptr addrspace(5) %"111", align 4
  %44 = load i32, ptr addrspace(5) %"98", align 4
  %"224" = add i32 %44, -32
  store i32 %"224", ptr addrspace(5) %"102", align 4
  %45 = load i32, ptr addrspace(5) %"99", align 4
  %"226" = add i32 %45, -32
  store i32 %"226", ptr addrspace(5) %"103", align 4
  %46 = load i32, ptr addrspace(5) %"100", align 4
  %"228" = add i32 %46, -32
  store i32 %"228", ptr addrspace(5) %"104", align 4
  %47 = load i32, ptr addrspace(5) %"101", align 4
  %"230" = add i32 %47, -32
  store i32 %"230", ptr addrspace(5) %"105", align 4
  %48 = load i32, ptr addrspace(5) %"111", align 4
  %49 = load i32, ptr addrspace(5) %"102", align 4
  %50 = icmp uge i32 %49, 32
  %51 = shl i32 %48, %49
  %"232" = select i1 %50, i32 0, i32 %51
  store i32 %"232", ptr addrspace(5) %"106", align 4
  %52 = load i32, ptr addrspace(5) %"111", align 4
  %53 = load i32, ptr addrspace(5) %"103", align 4
  %54 = icmp uge i32 %53, 32
  %55 = shl i32 %52, %53
  %"234" = select i1 %54, i32 0, i32 %55
  store i32 %"234", ptr addrspace(5) %"107", align 4
  %56 = load i32, ptr addrspace(5) %"111", align 4
  %57 = load i32, ptr addrspace(5) %"104", align 4
  %58 = icmp uge i32 %57, 32
  %59 = shl i32 %56, %57
  %"236" = select i1 %58, i32 0, i32 %59
  store i32 %"236", ptr addrspace(5) %"108", align 4
  %60 = load i32, ptr addrspace(5) %"111", align 4
  %61 = load i32, ptr addrspace(5) %"105", align 4
  %62 = icmp uge i32 %61, 32
  %63 = shl i32 %60, %61
  %"238" = select i1 %62, i32 0, i32 %63
  store i32 %"238", ptr addrspace(5) %"109", align 4
  store i32 0, ptr addrspace(5) %"110", align 4
  %64 = load i32, ptr addrspace(5) %"110", align 4
  %65 = load i32, ptr addrspace(5) %"106", align 4
  %"240" = add i32 %64, %65
  store i32 %"240", ptr addrspace(5) %"110", align 4
  %66 = load i32, ptr addrspace(5) %"110", align 4
  %67 = load i32, ptr addrspace(5) %"107", align 4
  %"243" = add i32 %66, %67
  store i32 %"243", ptr addrspace(5) %"110", align 4
  %68 = load i32, ptr addrspace(5) %"110", align 4
  %69 = load i32, ptr addrspace(5) %"108", align 4
  %"246" = add i32 %68, %69
  store i32 %"246", ptr addrspace(5) %"110", align 4
  %70 = load i32, ptr addrspace(5) %"110", align 4
  %71 = load i32, ptr addrspace(5) %"109", align 4
  %"249" = add i32 %70, %71
  store i32 %"249", ptr addrspace(5) %"110", align 4
  %72 = load i64, ptr addrspace(5) %"89", align 8
  %73 = load i32, ptr addrspace(5) %"110", align 4
  %"252" = inttoptr i64 %72 to ptr
  store i32 %73, ptr %"252", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
