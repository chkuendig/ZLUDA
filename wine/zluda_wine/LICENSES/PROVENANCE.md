# ZLUDA Wine bridge provenance and notices

Production is MIT OR Apache-2.0; retain root LICENSE-MIT, LICENSE-APACHE and LICENSES/ZLUDA-WoW64-MIT.txt. Original ZLUDA human authorship and design credits remain applicable.

Christian Kündig grants MIT OR Apache-2.0 for his copyright in these file-scoped owner contributions: wine/nvcuda/dlls/nvcuda/wow64/{pe.c,unix.c,wire.h}, wine/nvcuda/dlls/nvcuda/d3d11_cpu_copy.h and wine/zluda_wine/src/{wow64_local,wow64_modules,wow64_ops,wow64_pe,wow64_unix}.rs. The grant does not cover third-party portions or copied LGPL tests.

A fresh clean-room implementer replaced the mixed Wine/SveSop-derived production code using reviewed permissive ZLUDA declarations, cleared owner/scaffold reuse and source-free ABI/behavior facts. The reviewer knew the previous implementation; the supplied facts cite public Wine interfaces and retain that authority. The implementer did not open those URLs or read the removed implementations. Visible transcript audits, changed-file copying screens and CPU acceptance were recorded privately; these are engineering evidence, not proof of authorship. Retired per-thread callback records have process lifetime.

Copied LGPL oracles and the historical C reference were used separately for reviewer comparison, outside production compilation. Any distribution retaining those sources must preserve their full notices and corresponding source. The later clean branch removes them. Wine/Proton and registry dependencies keep their own licenses. Production consumes a separately reviewed, digest-pinned prepared-Wine ABI receipt and matching SDK binaries; it does not read Wine or oracle source. The private zluda-private knowledge/zluda-wine-bridge archive holds the full design, provenance and review receipts.
