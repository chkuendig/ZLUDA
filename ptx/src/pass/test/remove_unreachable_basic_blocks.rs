use crate::pass::*;

fn retained_functions(ptx: &str) -> String {
    let module = ptx_parser::parse_module_checked(ptx).unwrap();
    let mut resolver = GlobalStringIdentResolver2::new(SpirvWord(1));
    let mut scoped = ScopedResolver::new(&mut resolver);
    let directives = normalize_identifiers::run(&mut scoped, module.directives).unwrap();
    let directives = normalize_predicates::run(&mut resolver, directives).unwrap();
    let directives = expand_operands::run(&mut resolver, directives).unwrap();
    let directives = normalize_basic_blocks::run(&mut resolver, directives).unwrap();
    let directives = remove_unreachable_basic_blocks::run(directives).unwrap();
    directives
        .iter()
        .filter_map(|directive| match directive {
            Directive2::Method(function) => Some(
                resolver.ident_map[&function.name]
                    .name
                    .as_ref()
                    .unwrap()
                    .as_ref(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn kernel_rooted_calls() {
    let ptx = r#"
.version 7.0
.target sm_80
.address_size 64
.func b() { ret; }
.func a() { call b; ret; }
.func dead_callee() { ret; }
.func dead_caller() { call dead_callee; ret; }
.func dead_cycle_a();
.func dead_cycle_b() { call dead_cycle_a; ret; }
.func dead_cycle_a() { call dead_cycle_b; ret; }
.func recursive();
.func recursive() { call recursive; ret; }
.visible .entry kernel() { call a; call recursive; ret; }
"#;
    assert_eq!(retained_functions(ptx), "b a recursive recursive kernel");
}

#[test]
fn unreachable_call_does_not_keep_callee() {
    let ptx = r#"
.version 7.0
.target sm_80
.address_size 64
.func dead_callee() { ret; }
.visible .entry kernel() {
    bra done;
unused:
    call dead_callee;
done:
    ret;
}
"#;
    assert_eq!(retained_functions(ptx), "kernel");
}

#[test]
fn reachable_declarations_keep_source_order() {
    let ptx = r#"
.version 7.0
.target sm_80
.address_size 64
.extern .func live();
.extern .func dead();
.visible .entry kernel() { call live; ret; }
"#;
    assert_eq!(retained_functions(ptx), "live kernel");
}
