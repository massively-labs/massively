use cubecl::prelude::*;

use crate::{
    Error, Executor, MFlag, MIndex, MIter, MVal, Scalar,
    op::{PredicateOp, UnaryOp},
};

struct CountsEqual;

#[cubecl::cube]
impl UnaryOp<(MIndex, MIndex)> for CountsEqual {
    type Output = MFlag;

    fn apply(input: (MIndex, MIndex)) -> MFlag {
        crate::flag::from_bool(input.0 == input.1)
    }
}

struct CountNonZero;

#[cubecl::cube]
impl UnaryOp<MIndex> for CountNonZero {
    type Output = MFlag;

    fn apply(input: MIndex) -> MFlag {
        crate::flag::from_bool(input != 0)
    }
}

struct CountZero;

#[cubecl::cube]
impl UnaryOp<MIndex> for CountZero {
    type Output = MFlag;

    fn apply(input: MIndex) -> MFlag {
        crate::flag::from_bool(input == 0)
    }
}

struct IsSentinel;

#[cubecl::cube]
impl UnaryOp<MIndex> for IsSentinel {
    type Output = MFlag;

    fn apply(input: MIndex) -> MFlag {
        crate::flag::from_bool(input == MIndex::MAX)
    }
}

macro_rules! predicate_api {
    (
        $name:ident,
        $core_name:ident,
        $output:ty,
        |$exec:ident, $value:ident, $len:ident| $map:block,
        $doc:literal
    ) => {
        #[doc = $doc]
        pub fn $name<R, Input, Pred>(
            exec: &Executor<R>,
            input: Input,
            pred: Pred,
        ) -> Result<impl MVal<R, $output>, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Pred: PredicateOp<Input::Item>,
        {
            let len = input.capacity()?;
            let value = crate::predicate::$core_name(
                exec,
                crate::api::iter::lower_fixed::<R, _>(input),
                pred,
            )?;
            let $exec = exec;
            let $value = value;
            let $len = len;
            $map
        }
    };
}

/// Returns the logical number of items in the input as a device-resident value.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, MVal, vector::length};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
///
/// assert_eq!(length(&exec, input.slice(..)).unwrap().read(&exec).unwrap(), 4);
/// ```
pub fn length<R, Input>(
    exec: &Executor<R>,
    input: Input,
) -> Result<impl MVal<R, MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    let extent = input.logical_extent()?;
    let storage = extent.materialize(exec)?;
    Scalar::from_storage(storage)
}

predicate_api!(
    count_if,
    count_if,
    MIndex,
    |_exec, value, _len| { Ok(value) },
    r#"Counts items satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MVal, op, vector::count_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 2, 3, 4]);

assert_eq!(count_if(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), 2);
```
"#
);
predicate_api!(
    all_of,
    count_if,
    MFlag,
    |exec, value, len| {
        let output = crate::vector::map(
            exec,
            crate::zip2(value.as_iter(), crate::lazy::constant(len).take(1)),
            CountsEqual,
        )?;
        Scalar::from_storage(output)
    },
    r#"Returns whether every item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MVal, op, vector::all_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 6]);

assert!(massively::flag::is_set(
    all_of(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap()
));
```
"#
);
predicate_api!(
    any_of,
    count_if,
    MFlag,
    |exec, value, _len| {
        let output = crate::vector::map(exec, value.as_iter(), CountNonZero)?;
        Scalar::from_storage(output)
    },
    r#"Returns whether any item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MVal, op, vector::any_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4]);

assert!(massively::flag::is_set(
    any_of(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap()
));
```
"#
);
predicate_api!(
    none_of,
    count_if,
    MFlag,
    |exec, value, _len| {
        let output = crate::vector::map(exec, value.as_iter(), CountZero)?;
        Scalar::from_storage(output)
    },
    r#"Returns whether no item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MVal, op, vector::none_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 5]);

assert!(massively::flag::is_set(
    none_of(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap()
));
```
"#
);
predicate_api!(
    find_if,
    find_if,
    Option<MIndex>,
    |_exec, value, _len| { Ok(value.into_optional_index()) },
    r#"Returns the first index satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MVal, op, vector::find_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4, 6]);

assert_eq!(
    find_if(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(),
    Some(2)
);
```
"#
);
predicate_api!(
    is_partitioned,
    is_partitioned,
    MFlag,
    |exec, value, _len| {
        let output = crate::vector::map(exec, value.as_iter(), IsSentinel)?;
        Scalar::from_storage(output)
    },
    r#"Returns whether passing items precede failing items.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MVal, op, vector::is_partitioned};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 1, 3]);

assert!(massively::flag::is_set(
    is_partitioned(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap()
));
```
"#
);
