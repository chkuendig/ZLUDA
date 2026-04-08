define amdgpu_kernel void @histogram_accum(ptr addrspace(4) byref(i64) %"78", ptr addrspace(4) byref(i64) %"79") #0 {
  %"80" = alloca i64, align 8, addrspace(5)
  %"81" = alloca i64, align 8, addrspace(5)
  %"82" = alloca i32, align 4, addrspace(5)
  %"83" = alloca i32, align 4, addrspace(5)
  %"84" = alloca i32, align 4, addrspace(5)
  %"85" = alloca i32, align 4, addrspace(5)
  %"86" = alloca i32, align 4, addrspace(5)
  %"87" = alloca i32, align 4, addrspace(5)
  %"88" = alloca i32, align 4, addrspace(5)
  %"89" = alloca i32, align 4, addrspace(5)
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
  br label %1

1:                                                ; preds = %0
  br label %"77"

"77":                                             ; preds = %1
  %2 = load i64, ptr addrspace(4) %"78", align 8
  store i64 %2, ptr addrspace(5) %"80", align 8
  %3 = load i64, ptr addrspace(4) %"79", align 8
  store i64 %3, ptr addrspace(5) %"81", align 8
  %4 = load i64, ptr addrspace(5) %"80", align 8
  %"176" = inttoptr i64 %4 to ptr
  %5 = load i32, ptr %"176", align 4
  store i32 %5, ptr addrspace(5) %"82", align 4
  %6 = load i64, ptr addrspace(5) %"80", align 8
  %"177" = inttoptr i64 %6 to ptr
  %"57" = getelementptr inbounds i8, ptr %"177", i64 4
  %7 = load i32, ptr %"57", align 4
  store i32 %7, ptr addrspace(5) %"83", align 4
  %8 = load i64, ptr addrspace(5) %"80", align 8
  %"178" = inttoptr i64 %8 to ptr
  %"59" = getelementptr inbounds i8, ptr %"178", i64 8
  %9 = load i32, ptr %"59", align 4
  store i32 %9, ptr addrspace(5) %"84", align 4
  %10 = load i64, ptr addrspace(5) %"80", align 8
  %"179" = inttoptr i64 %10 to ptr
  %"61" = getelementptr inbounds i8, ptr %"179", i64 12
  %11 = load i32, ptr %"61", align 4
  store i32 %11, ptr addrspace(5) %"85", align 4
  store i32 0, ptr addrspace(5) %"100", align 4
  %12 = load i32, ptr addrspace(5) %"82", align 4
  %13 = load i32, ptr addrspace(5) %"100", align 4
  %14 = icmp uge i32 %13, 32
  %15 = lshr i32 %12, %13
  %"112" = select i1 %14, i32 0, i32 %15
  store i32 %"112", ptr addrspace(5) %"86", align 4
  %16 = load i32, ptr addrspace(5) %"83", align 4
  %17 = load i32, ptr addrspace(5) %"100", align 4
  %18 = icmp uge i32 %17, 32
  %19 = lshr i32 %16, %17
  %"115" = select i1 %18, i32 0, i32 %19
  store i32 %"115", ptr addrspace(5) %"87", align 4
  %20 = load i32, ptr addrspace(5) %"84", align 4
  %21 = load i32, ptr addrspace(5) %"100", align 4
  %22 = icmp uge i32 %21, 32
  %23 = lshr i32 %20, %21
  %"118" = select i1 %22, i32 0, i32 %23
  store i32 %"118", ptr addrspace(5) %"88", align 4
  %24 = load i32, ptr addrspace(5) %"85", align 4
  %25 = load i32, ptr addrspace(5) %"100", align 4
  %26 = icmp uge i32 %25, 32
  %27 = lshr i32 %24, %25
  %"121" = select i1 %26, i32 0, i32 %27
  store i32 %"121", ptr addrspace(5) %"89", align 4
  %28 = load i32, ptr addrspace(5) %"86", align 4
  %"180" = and i32 %28, 15
  store i32 %"180", ptr addrspace(5) %"86", align 4
  %29 = load i32, ptr addrspace(5) %"87", align 4
  %"182" = and i32 %29, 15
  store i32 %"182", ptr addrspace(5) %"87", align 4
  %30 = load i32, ptr addrspace(5) %"88", align 4
  %"184" = and i32 %30, 15
  store i32 %"184", ptr addrspace(5) %"88", align 4
  %31 = load i32, ptr addrspace(5) %"89", align 4
  %"186" = and i32 %31, 15
  store i32 %"186", ptr addrspace(5) %"89", align 4
  %32 = load i32, ptr addrspace(5) %"86", align 4
  %33 = shl i32 %32, 4
  %"188" = select i1 false, i32 0, i32 %33
  store i32 %"188", ptr addrspace(5) %"90", align 4
  %34 = load i32, ptr addrspace(5) %"87", align 4
  %35 = shl i32 %34, 4
  %"190" = select i1 false, i32 0, i32 %35
  store i32 %"190", ptr addrspace(5) %"91", align 4
  %36 = load i32, ptr addrspace(5) %"88", align 4
  %37 = shl i32 %36, 4
  %"192" = select i1 false, i32 0, i32 %37
  store i32 %"192", ptr addrspace(5) %"92", align 4
  %38 = load i32, ptr addrspace(5) %"89", align 4
  %39 = shl i32 %38, 4
  %"194" = select i1 false, i32 0, i32 %39
  store i32 %"194", ptr addrspace(5) %"93", align 4
  %40 = load i32, ptr addrspace(5) %"90", align 4
  %"196" = and i32 %40, 240
  store i32 %"196", ptr addrspace(5) %"90", align 4
  %41 = load i32, ptr addrspace(5) %"91", align 4
  %"198" = and i32 %41, 240
  store i32 %"198", ptr addrspace(5) %"91", align 4
  %42 = load i32, ptr addrspace(5) %"92", align 4
  %"200" = and i32 %42, 240
  store i32 %"200", ptr addrspace(5) %"92", align 4
  %43 = load i32, ptr addrspace(5) %"93", align 4
  %"202" = and i32 %43, 240
  store i32 %"202", ptr addrspace(5) %"93", align 4
  store i32 1, ptr addrspace(5) %"99", align 4
  %44 = load i32, ptr addrspace(5) %"99", align 4
  %45 = load i32, ptr addrspace(5) %"90", align 4
  %46 = icmp uge i32 %45, 32
  %47 = shl i32 %44, %45
  %"204" = select i1 %46, i32 0, i32 %47
  store i32 %"204", ptr addrspace(5) %"94", align 4
  %48 = load i32, ptr addrspace(5) %"99", align 4
  %49 = load i32, ptr addrspace(5) %"91", align 4
  %50 = icmp uge i32 %49, 32
  %51 = shl i32 %48, %49
  %"206" = select i1 %50, i32 0, i32 %51
  store i32 %"206", ptr addrspace(5) %"95", align 4
  %52 = load i32, ptr addrspace(5) %"99", align 4
  %53 = load i32, ptr addrspace(5) %"92", align 4
  %54 = icmp uge i32 %53, 32
  %55 = shl i32 %52, %53
  %"208" = select i1 %54, i32 0, i32 %55
  store i32 %"208", ptr addrspace(5) %"96", align 4
  %56 = load i32, ptr addrspace(5) %"99", align 4
  %57 = load i32, ptr addrspace(5) %"93", align 4
  %58 = icmp uge i32 %57, 32
  %59 = shl i32 %56, %57
  %"210" = select i1 %58, i32 0, i32 %59
  store i32 %"210", ptr addrspace(5) %"97", align 4
  store i32 0, ptr addrspace(5) %"98", align 4
  %60 = load i32, ptr addrspace(5) %"98", align 4
  %61 = load i32, ptr addrspace(5) %"94", align 4
  %"212" = add i32 %60, %61
  store i32 %"212", ptr addrspace(5) %"98", align 4
  %62 = load i32, ptr addrspace(5) %"98", align 4
  %63 = load i32, ptr addrspace(5) %"95", align 4
  %"215" = add i32 %62, %63
  store i32 %"215", ptr addrspace(5) %"98", align 4
  %64 = load i32, ptr addrspace(5) %"98", align 4
  %65 = load i32, ptr addrspace(5) %"96", align 4
  %"218" = add i32 %64, %65
  store i32 %"218", ptr addrspace(5) %"98", align 4
  %66 = load i32, ptr addrspace(5) %"98", align 4
  %67 = load i32, ptr addrspace(5) %"97", align 4
  %"221" = add i32 %66, %67
  store i32 %"221", ptr addrspace(5) %"98", align 4
  %68 = load i64, ptr addrspace(5) %"81", align 8
  %69 = load i32, ptr addrspace(5) %"98", align 4
  %"224" = inttoptr i64 %68 to ptr
  store i32 %69, ptr %"224", align 4
  fence seq_cst
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="preserve-sign" "denormal-fp-math-f32"="preserve-sign" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
