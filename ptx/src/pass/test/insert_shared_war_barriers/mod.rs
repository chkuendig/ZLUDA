use crate::pass::*;
use ptx_parser as ast;

// Runs the passes up to insert_shared_war_barriers and returns the order of shared loads
// ("ld"), shared stores ("st") and barriers ("bar") in the kernel body.
fn shared_accesses_and_barriers(ptx: &str) -> String {
    let module = match ptx_parser::parse_module_checked(ptx) {
        Ok(module) => module,
        Err(errors) => {
            for error in errors {
                eprintln!("{}", error);
            }
            panic!("PTX parse failed");
        }
    };
    let mut flat_resolver = GlobalStringIdentResolver2::new(SpirvWord(1));
    let mut scoped_resolver = ScopedResolver::new(&mut flat_resolver);
    let directives = normalize_identifiers::run(&mut scoped_resolver, module.directives).unwrap();
    let directives = normalize_predicates::run(&mut flat_resolver, directives).unwrap();
    let directives = expand_operands::run(&mut flat_resolver, directives).unwrap();
    let directives = normalize_basic_blocks::run(&mut flat_resolver, directives).unwrap();
    let directives = remove_unreachable_basic_blocks::run(directives).unwrap();
    let directives = insert_shared_war_barriers::run(&mut flat_resolver, directives).unwrap();
    let mut tokens = Vec::new();
    for directive in directives.iter() {
        if let Directive2::Method(Function {
            body: Some(body), ..
        }) = directive
        {
            for statement in body.iter() {
                match statement {
                    Statement::Instruction(ast::Instruction::Ld { data, .. })
                        if data.state_space == ast::StateSpace::Shared =>
                    {
                        tokens.push("ld")
                    }
                    Statement::Instruction(ast::Instruction::St { data, .. })
                        if data.state_space == ast::StateSpace::Shared =>
                    {
                        tokens.push("st")
                    }
                    Statement::Instruction(ast::Instruction::Bar { .. }) => tokens.push("bar"),
                    _ => {}
                }
            }
        }
    }
    tokens.join(" ")
}

#[test]
fn downsweep_value_staging() {
    // One barrier, before the first value-staging store.
    assert_eq!(
        shared_accesses_and_barriers(include_str!("downsweep_value_staging.ptx")),
        "st st bar ld ld ld bar st st ld ld bar st st bar"
    );
}

#[test]
fn partial_tile_value_staging() {
    // The pending exchange read reaches the staging block through both paths.
    assert_eq!(
        shared_accesses_and_barriers(include_str!("partial_tile_value_staging.ptx")),
        "bar ld bar st ld bar"
    );
}

#[test]
fn no_barrier_in_block() {
    assert_eq!(
        shared_accesses_and_barriers(include_str!("no_barrier_in_block.ptx")),
        "bar ld st"
    );
}

#[test]
fn exempt_accesses() {
    assert_eq!(
        shared_accesses_and_barriers(include_str!("exempt_accesses.ptx")),
        "bar ld st ld st ld st bar"
    );
}
