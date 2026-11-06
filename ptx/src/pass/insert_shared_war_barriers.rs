use super::*;
use rustc_hash::FxHashSet;

// Inserts `bar.sync 0` before a shared-memory store that may overwrite a slot other
// threads of the CTA can still be reading since the last barrier (write-after-read), or
// that another thread's volatile store may still overwrite (write-after-write).
//
// CUB 1.0.2's radix-sort downsweep, built into NVIDIA's PhysX/APEX modules, reads the
// key exchange buffer back after a barrier and then stages the values through the same
// temp_storage union with no barrier in between (CUB 1.8 opens GatherScatterValues with
// CTA_SYNC()). On NVIDIA the value load latency hides the race. On RDNA a delayed wave
// can read a staged value in place of a key.
//
// The rule, per function, on the basic blocks built by normalize_basic_blocks:
// * Only weak ld.shared/st.shared take part. .volatile and scoped accesses are the
//   warp-synchronous idiom and are left alone (except volatile stores, see below).
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
//   whose own barrier does not hang.
//
// Write-after-write. The CUDPP scan4 kernels in NVIDIA's PhysX 2.x runtimes zero-pad
// their warp-scan window with volatile stores right after a barrier. Lane 31 of each
// warp then stores the warp total into the same array, in a block of its own, with no
// barrier in between, so a delayed warp 0 can zero the totals of other warps.
// * A volatile st.shared is pending from the moment it executes until the next barrier,
//   like a weak load. It is never checked itself. Volatile loads stay exempt: in the
//   warp-scan idiom each follows a volatile store into the same window, which is
//   already pending. Weak stores do not become pending (no write-after-write between
//   two weak stores).
// * A weak store conflicts with a pending volatile store under the same address and
//   variable rules as with a pending load.
// * In a block with an aligned `bar.sync 0` the barrier goes before the store as above.
//   Otherwise it goes at the end of a block the conflicting volatile stores were issued
//   in, before its terminator, when exactly one such block qualifies (block_end_is_safe):
//   it contains an aligned `bar.sync 0` and all the stores come from it, or it closes a
//   barrier-free region that is entered only through one block with an aligned
//   `bar.sync 0` and left only through it, and all the stores come from that region.
//   A pending access was issued after the last barrier on its path, so this point lies
//   between the two stores on every path. Every thread executes it exactly once per
//   execution of the existing barrier's block, so it cannot hang a kernel whose own
//   barrier does not hang. Otherwise nothing is inserted.
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
        let successors = successors
            .into_iter()
            .filter_map(|label| labels.get(&label).copied())
            .collect::<Vec<_>>();
        blocks.push(Block {
            events: function.events(&body[start..end], start)?,
            eligible: body[start..end].iter().any(|s| function.is_full_barrier(s)),
            // Only blocks ending in a branch pass pending accesses on, so only they can
            // receive a barrier before their terminator.
            terminator: (!successors.is_empty()).then_some(end - 1),
            successors,
            predecessors: Vec::new(),
        });
    }
    for i in 0..blocks.len() {
        for j in 0..blocks[i].successors.len() {
            let successor = blocks[i].successors[j];
            blocks[successor].predecessors.push(i);
        }
    }
    // Barriers at the end of a block (write-after-write fall-back). They only ever remove
    // pending accesses, so the set grows monotonically to a fixpoint.
    let mut block_ends = FxHashSet::<usize>::default();
    let mut inserts;
    loop {
        // Forward may-analysis of pending accesses at block entry.
        let mut entry = vec![FxHashSet::<Pending>::default(); blocks.len()];
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..blocks.len() {
                let exit = function.walk(
                    &blocks,
                    i,
                    &entry[i],
                    &block_ends,
                    &mut Vec::new(),
                    &mut FxHashSet::default(),
                );
                for &successor in blocks[i].successors.iter() {
                    for pending in exit.iter() {
                        changed |= entry[successor].insert(*pending);
                    }
                }
            }
        }
        inserts = Vec::new();
        let mut new_block_ends = FxHashSet::default();
        for i in 0..blocks.len() {
            function.walk(
                &blocks,
                i,
                &entry[i],
                &block_ends,
                &mut inserts,
                &mut new_block_ends,
            );
        }
        let known = block_ends.len();
        block_ends.extend(new_block_ends);
        if block_ends.len() == known {
            break;
        }
    }
    inserts.extend(block_ends.iter().filter_map(|&i| blocks[i].terminator));
    if inserts.is_empty() {
        return Ok(());
    }
    inserts.sort_unstable();
    inserts.dedup();
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

// An access still pending: the .shared variable it points into, the block it was issued
// in, and whether it is a volatile store (otherwise a weak load or a call).
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Pending {
    root: Root,
    block: usize,
    store: bool,
}

enum Event {
    Barrier,
    Call,
    Load(Option<Address>, Root),
    // index of the store in the function body
    Store(usize, Option<Address>, Root),
    VolatileStore(Option<Address>, Root),
}

struct Block {
    events: Vec<Event>,
    eligible: bool,
    // body index of the final branch
    terminator: Option<usize>,
    successors: Vec<usize>,
    predecessors: Vec<usize>,
}

impl Block {
    // Contains a barrier of any kind or a call.
    fn synchronizes(&self) -> bool {
        self.events
            .iter()
            .any(|event| matches!(event, Event::Barrier | Event::Call))
    }
}

// The block that gets a barrier before its terminator for a weak store in block `store`
// that conflicts with pending volatile stores issued in `origins`, if exactly one of
// them qualifies.
fn block_end_for(blocks: &[Block], origins: &FxHashSet<usize>, store: usize) -> Option<usize> {
    let mut found = None;
    for &candidate in origins.iter() {
        if block_end_is_safe(blocks, candidate, origins, store) {
            if found.is_some() {
                return None;
            }
            found = Some(candidate);
        }
    }
    found
}

// A barrier at the end of block `end` is safe when every thread executes `end` exactly
// once after each execution of one block `head` with an aligned `bar.sync 0` and before
// the next, with no other barrier in between, and when it lies after every pending store
// from `origins` on each path to the store:
// * `end` has such a barrier itself, and all origins are `end`; or
// * walking back from `end` through predecessors, stopping at blocks with such a
//   barrier, reaches exactly one of them (`head`) and never the entry block, `end` or the
//   store's block. The blocks passed on the way and `end` contain no barrier and no call,
//   and each of them ends in a branch. Every successor of `head` and of the blocks passed
//   is one of the blocks passed or `end`. All origins are `head`, `end` or a block passed.
fn block_end_is_safe(
    blocks: &[Block],
    end: usize,
    origins: &FxHashSet<usize>,
    store: usize,
) -> bool {
    let block = &blocks[end];
    if end == store || block.terminator.is_none() {
        return false;
    }
    if block.eligible {
        return origins.iter().all(|&origin| origin == end);
    }
    if block.synchronizes() {
        return false;
    }
    let mut passed = FxHashSet::default();
    let mut heads = FxHashSet::default();
    let mut todo = block.predecessors.clone();
    while let Some(i) = todo.pop() {
        if passed.contains(&i) || heads.contains(&i) {
            continue;
        }
        let other = &blocks[i];
        if other.eligible {
            heads.insert(i);
            continue;
        }
        if i == 0 || i == end || i == store || other.synchronizes() || other.terminator.is_none() {
            return false;
        }
        passed.insert(i);
        todo.extend(other.predecessors.iter().copied());
    }
    if heads.len() != 1 {
        return false;
    }
    let head = heads.into_iter().next().unwrap();
    passed
        .iter()
        .chain(std::iter::once(&head))
        .all(|&i| {
            blocks[i]
                .successors
                .iter()
                .all(|successor| *successor == end || passed.contains(successor))
        })
        && origins
            .iter()
            .all(|origin| *origin == end || *origin == head || passed.contains(origin))
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
                Statement::Instruction(ast::Instruction::St { data, arguments })
                    if data.qualifier == ast::LdStQualifier::Volatile
                        && is_shared(data.state_space) =>
                {
                    let (address, root) = self.address(arguments.src1, &versions);
                    events.push(Event::VolatileStore(address, root));
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

    // Walks block `index` and returns the accesses still pending at its exit. Appends the
    // body indices of stores that need a barrier in front of them, and the blocks that
    // need a barrier before their terminator. Blocks in `block_ends` already have one.
    fn walk(
        &self,
        blocks: &[Block],
        index: usize,
        entry: &FxHashSet<Pending>,
        block_ends: &FxHashSet<usize>,
        inserts: &mut Vec<usize>,
        new_block_ends: &mut FxHashSet<usize>,
    ) -> FxHashSet<Pending> {
        let block = &blocks[index];
        let mut pending = entry
            .iter()
            .map(|access| (None, *access))
            .collect::<Vec<(Option<Address>, Pending)>>();
        let issued = |root, store| Pending {
            root,
            block: index,
            store,
        };
        for (i, event) in block.events.iter().enumerate() {
            match *event {
                Event::Barrier => pending.clear(),
                Event::Call => pending.push((None, issued(Root::Unknown, false))),
                Event::Load(address, root) => {
                    let owned = address.is_some()
                        && block.events[i + 1..]
                            .iter()
                            .take_while(|event| !matches!(event, Event::Barrier))
                            .any(|event| {
                                matches!(event, Event::Store(_, store, _) if *store == address)
                            });
                    if !owned {
                        pending.push((address, issued(root, false)));
                    }
                }
                Event::VolatileStore(address, root) => {
                    pending.push((address, issued(root, true)));
                }
                Event::Store(store_index, address, root) => {
                    let mut conflicts = pending.iter().filter(|(pending_address, access)| {
                        (pending_address.is_none() || *pending_address != address)
                            && self.may_alias(access.root, root)
                    });
                    if block.eligible {
                        if conflicts.next().is_some() {
                            inserts.push(store_index);
                            pending.clear();
                        }
                        continue;
                    }
                    // Write-after-write fall-back: a barrier at the end of a block the
                    // conflicting volatile stores came from.
                    let origins = conflicts
                        .filter(|(_, access)| access.store)
                        .map(|(_, access)| access.block)
                        .collect::<FxHashSet<_>>();
                    if let Some(block_end) = block_end_for(blocks, &origins, index) {
                        new_block_ends.insert(block_end);
                    }
                }
            }
        }
        if block_ends.contains(&index) {
            pending.clear();
        }
        pending.into_iter().map(|(_, access)| access).collect()
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
