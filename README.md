# ZLUDA Wine CUDA component

This independent MIT OR Apache-2.0 component supplies the PE64, PE32 and Unix
Wine CUDA bridge. Its clean-room production source was extracted from public
ZLUDA `8a8e3b76`; existing notices, owner grants and human credits are retained.
See `wine/zluda_wine/LICENSES/PROVENANCE.md` and the root license texts.

Set `ZLUDA_WINE_BACKEND_SOURCE` to an absolute checkout of the backend commit
in `BACKEND-SOURCE.json`. Generation byte-checks its six permissive source
inputs. There are no backend Cargo dependencies or duplicated declarations.
Proton pins this component and the backend independently and supplies the
matching prepared Wine SDK and digest-pinned ABI receipt. Build all three
modules together using `wine/zluda_wine/build-bridge.sh`; do not mix targets.
The production source capsule includes the admitted backend input closure and
can rebuild all three targets offline without the original checkouts.

Optional PE32 `ZLUDA_WINE_CALL_LOG` tracing is retained at the owner's request.
It is disabled by default. `wine/w1-fixtures/tools/cuda-count` is a test-only
counter, built exclusively into independent fixture output; it is not a
shipping bridge input and must not be installed in the Proton product.

The temporary public staging ref lives in chkuendig/ZLUDA. A later dedicated
repository must receive these same commits without rewriting their history.
