//! Narrow hardware policy, evaluated before HIP initialization.

use std::path::Path;

fn matches_deck(vendor: &str, product: &str, board: &str, targets: &[u32]) -> bool {
    vendor.trim() == "Valve"
        && [product.trim(), board.trim()]
            .iter()
            .any(|model| matches!(*model, "Jupiter" | "Galileo"))
        && targets == [100303]
}

pub(super) fn is_deck_apu() -> bool {
    detect(Path::new("/sys")).unwrap_or(false)
}

fn detect(root: &Path) -> Option<bool> {
    let dmi = root.join("class/dmi/id");
    let vendor = std::fs::read_to_string(dmi.join("sys_vendor")).ok()?;
    let product = std::fs::read_to_string(dmi.join("product_name")).ok()?;
    let board = std::fs::read_to_string(dmi.join("board_name")).ok()?;
    let mut targets = Vec::new();
    for node in std::fs::read_dir(root.join("class/kfd/kfd/topology/nodes")).ok()? {
        let text = std::fs::read_to_string(node.ok()?.path().join("properties")).ok()?;
        let value = text.lines().find_map(|line| line.strip_prefix("gfx_target_version "))?;
        let target = value.trim().parse::<u32>().ok()?;
        if target != 0 {
            targets.push(target);
        }
    }
    Some(matches_deck(&vendor, &product, &board, &targets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_models_require_the_deck_apu() {
        for model in ["Jupiter", "Galileo"] {
            assert!(matches_deck("Valve\n", model, model, &[100303]));
            assert!(!matches_deck("Valve", model, model, &[100304]));
            assert!(!matches_deck("Valve", model, model, &[]));
        }
    }

    #[test]
    fn other_hosts_and_attached_gpus_are_excluded() {
        assert!(!matches_deck("ASUS", "ROG Ally", "ROG Ally", &[100303]));
        assert!(!matches_deck("Valve", "Other", "Other", &[100303]));
        assert!(!matches_deck("Valve", "Galileo", "Galileo", &[100303, 110000]));
    }
}
