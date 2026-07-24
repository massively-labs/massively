use cubecl::prelude::Runtime;

use crate::{Error, Executor, MFlag, MIndex, MIter, op::PredicateOp};

macro_rules! predicate_api {
    ($name:ident, $core_name:ident, $device_output:ty, $output:ty, $map:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Input, Pred>(
            exec: &Executor<R>,
            input: Input,
            pred: Pred,
        ) -> Result<$output, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Pred: PredicateOp<Input::Item>,
        {
            let len = input.len()?;
            let value = crate::predicate::$core_name(
                exec,
                crate::api::iter::lower_fixed::<R, _>(input),
                pred,
            )?;
            let value: $device_output = value.read(exec)?;
            Ok(($map)(value, len))
        }
    };
}

predicate_api!(
    count_if,
    count_if,
    MIndex,
    MIndex,
    |value, _len| value,
    r#"Counts items satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::count_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 2, 3, 4]);

assert_eq!(count_if(&exec, input.slice(..), Even).unwrap(), 2);
```
"#
);
predicate_api!(
    all_of,
    count_if,
    u32,
    MFlag,
    |value, len| crate::flag::from_bool(value == len),
    r#"Returns whether every item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::all_of};

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
    all_of(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
predicate_api!(
    any_of,
    count_if,
    u32,
    MFlag,
    |value, _len| crate::flag::from_bool(value != 0),
    r#"Returns whether any item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::any_of};

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
    any_of(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
predicate_api!(
    none_of,
    count_if,
    u32,
    MFlag,
    |value, _len| crate::flag::from_bool(value == 0),
    r#"Returns whether no item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::none_of};

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
    none_of(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
predicate_api!(
    find_if,
    find_if,
    MIndex,
    Option<MIndex>,
    |index, _len| (index != u32::MAX).then_some(index),
    r#"Returns the first index satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::find_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4, 6]);

assert_eq!(find_if(&exec, input.slice(..), Even).unwrap(), Some(2));
```
"#
);
predicate_api!(
    is_partitioned,
    is_partitioned,
    u32,
    MFlag,
    |index, _len| crate::flag::from_bool(index == u32::MAX),
    r#"Returns whether passing items precede failing items.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::is_partitioned};

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
    is_partitioned(&exec, input.slice(..), Even).unwrap()
));
```
"#
);
