use super::*;
use rustc_hash::FxHashSet;

// Inserts `bar.sync 0` before a shared-memory store that may overwrite a slot other
// threads of the CTA can still be reading since the last barrier (write-after-read).
//
// CUB 1.0.2's radix-sort downsweep, built into NVIDIA's PhysX/APEX modules, reads the
// key exchange buffer back after a barrier and then stages the values through the same
// temp_storage union with no barrier in between (CUB 1.8 opens GatherScatterValues with
// CTA_SYNC()). On NVIDIA the value load latency hides the race. On RDNA a delayed wave
// can read a staged value in place of a key.
//
// The rule, per function, on the basic blocks built by normalize_basic_blocks:
// * Only weak ld.shared/st.shared take part. .volatile and scoped accesses are the
//   warp-synchronous idiom and are left alone.
// * A load is pending from the moment it executes until the next barrier, along every
//   path through the CFG. Inside a block it keeps its address (base register, version
//   of that register, immediate offset); across blocks only the .shared variable it
//   points into (or "unknown").
// * A load whose exact address the same thread stores to later in the same block and
//   barrier interval is a read-modify-write of its own slot. It never becomes pending:
//   a second thread touching that slot would already race on the store.
// * A store conflicts with a pending load unless both use the same address, or they
//   point into different .shared variables (two .extern variables may alias).
// * `bar.sync 0` goes before the first conflicting store, and only in a block that
//   already contains an aligned `bar.sync 0` without thread count. Every thread that
//   enters such a block executes all of it, so the new barrier is reached by exactly
//   the threads that reach the existing one, once per visit. It cannot hang a kernel
//   whose own barrier does not hang. Other blocks are never changed.
pub(super) fn run<'input>(
    resolver: &mut GlobalStringIdentResolver2<'input>,
    mut directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>, TranslateError> {
    // .shared variable -> declared .extern
    let mut variables = FxHashMap::<SpirvWord, bool>::default();
    for directive in directives.iter() {
        if let Directive2::Variable(linking, var) = directive {
            if is_shared(var.info.state_space) {
                variables.insert(var.name, linking.contains(ast::LinkingDirective::EXTERN));
            }
        }
    }
    for directive in directives.iter_mut() {
        if let Directive2::Method(Function {
            body: Some(body), ..
        }) = directive
        {
            run_method(resolver, variables.clone(), body)?;
        }
    }
    Ok(directives)
}

fn run_method(
    resolver: &mut GlobalStringIdentResolver2<'_>,
    variables: FxHashMap<SpirvWord, bool>,
    body: &mut Vec<Statement<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<(), TranslateError> {
    let function = FunctionInfo::new(variables, &body[..])?;
    let mut labels = FxHashMap::default();
    let mut starts = Vec::new();
    for (index, statement) in body.iter().enumerate() {
        if let Statement::Label(label) = statement {
            labels.insert(*label, starts.len());
            starts.push(index);
        }
    }
    if starts.first() != Some(&0) {
        return Ok(());
    }
    let mut blocks = Vec::with_capacity(starts.len());
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(body.len());
        let successors = match &body[end - 1] {
            Statement::Instruction(ast::Instruction::Bra { arguments }) => vec![arguments.src],
            Statement::Conditional(branch) => vec![branch.if_true, branch.if_false],
            _ => Vec::new(),
        };
        blocks.push(Block {
            events: function.events(&body[start..end], start)?,
            eligible: body[start..end].iter().any(|s| function.is_full_barrier(s)),
            successors: successors
                .into_iter()
                .filter_map(|label| labels.get(&label).copied())
                .collect(),
        });
    }
    // Forward may-analysis of pending loads at block entry.
    let mut entry = vec![FxHashSet::<Root>::default(); blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (i, block) in blocks.iter().enumerate() {
            let exit = function.walk(block, &entry[i], &mut Vec::new());
            for &successor in block.successors.iter() {
                for root in exit.iter() {
                    changed |= entry[successor].insert(*root);
                }
            }
        }
    }
    let mut inserts = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        function.walk(block, &entry[i], &mut inserts);
    }
    if inserts.is_empty() {
        return Ok(());
    }
    inserts.sort_unstable();
    let mut inserts = inserts.into_iter().peekable();
    let old_body = std::mem::take(body);
    for (index, statement) in old_body.into_iter().enumerate() {
        if inserts.next_if_eq(&index).is_some() {
            let zero = resolver.register_unnamed(Some((
                ast::Type::Scalar(ast::ScalarType::U32),
                ast::StateSpace::Reg,
            )));
            body.push(Statement::Constant(ConstantDefinition {
                dst: zero,
                typ: ast::ScalarType::U32,
                value: ast::ImmediateValue::U64(0),
            }));
            body.push(Statement::Instruction(ast::Instruction::Bar {
                data: ast::BarData { aligned: true },
                arguments: ast::BarArgs {
                    src1: zero,
                    src2: None,
                },
            }));
        }
        body.push(statement);
    }
    Ok(())
}

fn is_shared(space: ast::StateSpace) -> bool {
    matches!(space, ast::StateSpace::Shared | ast::StateSpace::SharedCta)
}

// The .shared variable an address points into, if its definitions prove one.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Root {
    Variable(SpirvWord),
    Unknown,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct Address {
    base: SpirvWord,
    version: u32,
    offset: i64,
}

enum Origin {
    Copy(SpirvWord),
    Sum(SpirvWord, SpirvWord),
    Difference(SpirvWord),
    Element(SpirvWord, SpirvWord),
    Other,
}

enum Event {
    Barrier,
    Call,
    Load(Option<Address>, Root),
    // index of the store in the function body
    Store(usize, Option<Address>, Root),
}

struct Block {
    events: Vec<Event>,
    eligible: bool,
    successors: Vec<usize>,
}

struct FunctionInfo {
    variables: FxHashMap<SpirvWord, bool>,
    constants: FxHashMap<SpirvWord, u64>,
    // None: defined more than once
    definitions: FxHashMap<SpirvWord, Option<Origin>>,
}

impl FunctionInfo {
    fn new(
        mut variables: FxHashMap<SpirvWord, bool>,
        body: &[Statement<ast::Instruction<SpirvWord>, SpirvWord>],
    ) -> Result<Self, TranslateError> {
        let mut constants = FxHashMap::<SpirvWord, u64>::default();
        let mut definitions = FxHashMap::<SpirvWord, Option<Origin>>::default();
        for statement in body {
            match statement {
                Statement::Variable(var) if is_shared(var.info.state_space) => {
                    variables.insert(var.name, false);
                }
                Statement::Constant(constant) => {
                    if let Some(value) = constant.value.as_u64() {
                        constants.insert(constant.dst, value);
                    }
                }
                _ => {}
            }
            for_each_definition(statement, |word| {
                definitions
                    .entry(word)
                    .and_modify(|existing| *existing = None)
                    .or_insert_with(|| Some(origin(statement)));
            })?;
        }
        Ok(Self {
            variables,
            constants,
            definitions,
        })
    }

    fn is_full_barrier(
        &self,
        statement: &Statement<ast::Instruction<SpirvWord>, SpirvWord>,
    ) -> bool {
        match statement {
            Statement::Instruction(ast::Instruction::Bar { data, arguments }) => {
                data.aligned
                    && arguments.src2.is_none()
                    && self.constants.get(&arguments.src1) == Some(&0)
            }
            _ => false,
        }
    }

    fn variable_of(&self, word: SpirvWord, depth: u32) -> Option<SpirvWord> {
        if self.variables.contains_key(&word) {
            return Some(word);
        }
        if depth == 16 {
            return None;
        }
        match self.definitions.get(&word)? {
            Some(Origin::Copy(src))
            | Some(Origin::Difference(src))
            | Some(Origin::Element(src, _)) => self.variable_of(*src, depth + 1),
            Some(Origin::Sum(src1, src2)) => match (
                self.variable_of(*src1, depth + 1),
                self.variable_of(*src2, depth + 1),
            ) {
                (Some(var), None) | (None, Some(var)) => Some(var),
                _ => None,
            },
            _ => None,
        }
    }

    fn may_alias(&self, root1: Root, root2: Root) -> bool {
        match (root1, root2) {
            (Root::Variable(var1), Root::Variable(var2)) => {
                var1 == var2 || (self.variables[&var1] && self.variables[&var2])
            }
            _ => true,
        }
    }

    fn events(
        &self,
        statements: &[Statement<ast::Instruction<SpirvWord>, SpirvWord>],
        first_index: usize,
    ) -> Result<Vec<Event>, TranslateError> {
        let mut versions = FxHashMap::<SpirvWord, u32>::default();
        let mut events = Vec::new();
        for (i, statement) in statements.iter().enumerate() {
            match statement {
                Statement::Instruction(ast::Instruction::Bar { .. })
                | Statement::Instruction(ast::Instruction::BarRed { .. }) => {
                    events.push(Event::Barrier)
                }
                Statement::Instruction(ast::Instruction::Call { arguments, .. })
                    if !arguments.is_external =>
                {
                    events.push(Event::Call)
                }
                Statement::Instruction(ast::Instruction::Ld { data, arguments })
                    if is_weak_shared(data.qualifier, data.state_space) =>
                {
                    let (address, root) = self.address(arguments.src, &versions);
                    events.push(Event::Load(address, root));
                }
                Statement::Instruction(ast::Instruction::St { data, arguments })
                    if is_weak_shared(data.qualifier, data.state_space) =>
                {
                    let (address, root) = self.address(arguments.src1, &versions);
                    events.push(Event::Store(first_index + i, address, root));
                }
                _ => {}
            }
            for_each_definition(statement, |word| *versions.entry(word).or_insert(0) += 1)?;
        }
        Ok(events)
    }

    fn address(
        &self,
        word: SpirvWord,
        versions: &FxHashMap<SpirvWord, u32>,
    ) -> (Option<Address>, Root) {
        let (base, offset) = match self.definitions.get(&word) {
            Some(Some(Origin::Element(base, offset))) => match self.constants.get(offset) {
                Some(offset) => (*base, *offset as i64),
                None => return (None, self.root(word)),
            },
            _ => (word, 0),
        };
        let address = Address {
            base,
            version: versions.get(&base).copied().unwrap_or(0),
            offset,
        };
        (Some(address), self.root(base))
    }

    fn root(&self, word: SpirvWord) -> Root {
        match self.variable_of(word, 0) {
            Some(var) => Root::Variable(var),
            None => Root::Unknown,
        }
    }

    // Returns the roots of loads still pending at block exit and appends the body
    // indices of stores that need a barrier.
    fn walk(
        &self,
        block: &Block,
        entry: &FxHashSet<Root>,
        inserts: &mut Vec<usize>,
    ) -> FxHashSet<Root> {
        let mut pending = entry
            .iter()
            .map(|root| (None, *root))
            .collect::<Vec<(Option<Address>, Root)>>();
        for (i, event) in block.events.iter().enumerate() {
            match *event {
                Event::Barrier => pending.clear(),
                Event::Call => pending.push((None, Root::Unknown)),
                Event::Load(address, root) => {
                    let owned = address.is_some()
                        && block.events[i + 1..]
                            .iter()
                            .take_while(|event| !matches!(event, Event::Barrier))
                            .any(|event| {
                                matches!(event, Event::Store(_, store, _) if *store == address)
                            });
                    if !owned {
                        pending.push((address, root));
                    }
                }
                Event::Store(index, address, root) => {
                    if block.eligible
                        && pending.iter().any(|(load, load_root)| {
                            (load.is_none() || *load != address)
                                && self.may_alias(*load_root, root)
                        })
                    {
                        inserts.push(index);
                        pending.clear();
                    }
                }
            }
        }
        pending.into_iter().map(|(_, root)| root).collect()
    }
}

fn is_weak_shared(qualifier: ast::LdStQualifier, space: ast::StateSpace) -> bool {
    qualifier == ast::LdStQualifier::Weak && is_shared(space)
}

fn origin(statement: &Statement<ast::Instruction<SpirvWord>, SpirvWord>) -> Origin {
    match statement {
        Statement::Instruction(ast::Instruction::Mov { arguments, .. }) => {
            Origin::Copy(arguments.src)
        }
        Statement::Instruction(ast::Instruction::Add { arguments, .. }) => {
            Origin::Sum(arguments.src1, arguments.src2)
        }
        Statement::Instruction(ast::Instruction::Sub { arguments, .. }) => {
            Origin::Difference(arguments.src1)
        }
        Statement::PtrAccess(PtrAccess {
            ptr_src,
            offset_src,
            ..
        }) => Origin::Element(*ptr_src, *offset_src),
        _ => Origin::Other,
    }
}

fn for_each_definition(
    statement: &Statement<ast::Instruction<SpirvWord>, SpirvWord>,
    mut f: impl FnMut(SpirvWord),
) -> Result<(), TranslateError> {
    match statement {
        Statement::Instruction(instruction) => ast::visit(
            instruction,
            &mut |word: &SpirvWord,
                  _: Option<(&ptx_parser::Type, ptx_parser::StateSpace)>,
                  is_dst: bool,
                  _: bool| {
                if is_dst {
                    f(*word);
                }
                Ok::<_, TranslateError>(())
            },
        ),
        Statement::Constant(ConstantDefinition { dst, .. })
        | Statement::Conversion(ImplicitConversion { dst, .. })
        | Statement::PtrAccess(PtrAccess { dst, .. })
        | Statement::FunctionPointer(FunctionPointerDetails { dst, .. })
        | Statement::VectorRead(VectorRead { scalar_dst: dst, .. })
        | Statement::VectorWrite(VectorWrite { vector_dst: dst, .. })
        | Statement::FpSaturate { dst, .. } => {
            f(*dst);
            Ok(())
        }
        Statement::RepackVector(RepackVectorDetails {
            packed, unpacked, ..
        }) => {
            f(*packed);
            for word in unpacked.iter().flatten() {
                f(*word);
            }
            Ok(())
        }
        Statement::Label(..)
        | Statement::Variable(..)
        | Statement::Conditional(..)
        | Statement::RetValue(..)
        | Statement::SetMode(..)
        | Statement::FpModeRequired { .. } => Ok(()),
    }
}
