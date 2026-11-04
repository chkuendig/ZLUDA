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

#[test]
fn fallback_not_unique() {
    assert_eq!(
        shared_accesses_and_barriers(include_str!("fallback_not_unique.ptx")),
        "bar st bar st st"
    );
}
