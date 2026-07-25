#!/usr/bin/env bash
set -euo pipefail

forbidden='MExtent|MSequence|Iteration|SegmentKeyInput|SegmentedValues|SegmentContextRead|SegmentRead|CanonicalAlloc|CanonicalStorage|CanonicalAbi|ScratchAbi|SortAbi|ItemDispatch|MutableItem|MutableDispatch|ConcreteOutput|OutputOperation|RadixKeyAbi|RadixStorage|LayoutCompatible|AllocColumns|RowAlloc|ScratchStorage|KernelRow|GatherInput|KernelInput|IterLength|SliceExpression|ReadExpression|OutputExpression|StorageLayout|MaterializeDispatch|ReduceDispatch|ReadArity|StorageArity|Eval[1-8]|\bColumn\b'
rejected_public_abstractions='MValue|MExtent|MSequence|Iteration|SegmentContexts'

if rg -n -g '*.html' "$rejected_public_abstractions" target/doc/massively; then
    echo 'rejected abstractions leaked into the public API documentation' >&2
    exit 1
fi

mapfile -d '' pages < <(
    find target/doc/massively -type f \
        \( -name 'fn.*.html' -o -name 'trait.MRadix.html' \
        -o -name 'trait.MAlloc.html' \
        -o -name 'trait.MVal.html' \
        -o -name 'trait.MIter.html' -o -name 'trait.MIterMut.html' \
        -o -name 'trait.MStorage.html' \) \
        -print0
)

if rg -n "$forbidden" "${pages[@]}"; then
    echo 'internal kernel constraints leaked into the public API documentation' >&2
    exit 1
fi

legacy_pages=(
    trait.CanonicalForm.html trait.WritableFrom.html trait.ToCanonical.html
    trait.MItem.html trait.MScratchItem.html trait.MSortItem.html trait.MutableItem.html
    trait.RadixKey.html
    struct.Zip.html struct.SegmentRead.html struct.SegmentContextRead.html
    struct.SegmentContexts.html
    struct.MExtent.html
    trait.MValue.html trait.MSequence.html type.MBool.html
)

for page in "${legacy_pages[@]}"; do
    test -z "$(find target/doc/massively -type f -name "$page" -print -quit)"
done

test -f "target/doc/massively/trait.MVal.html"
test -z "$(find target/doc/massively -type f -name 'struct.Scalar.html' -print -quit)"
test -z "$(find target/doc/massively -type f -name 'trait.MExact.html' -print -quit)"
test -z "$(find target/doc/massively -type f -name 'fn.into_exact.html' -print -quit)"
test -z "$(find target/doc/massively -type f \( -name 'struct.LogicalLen.html' -o -name 'struct.LogicalLenIter.html' -o -name 'enum.LogicalExtent.html' \) -print -quit)"
test ! -e "target/doc/massively/struct.HostVal.html"
test ! -e "target/doc/massively/struct.DeviceVal.html"
test ! -e "target/doc/massively/struct.MVal.html"

for arity in {2..12}; do
    test -z "$(find target/doc/massively -type f -name "fn.unzip${arity}.html" -print -quit)"
    test -z "$(find target/doc/massively -type f -name "fn.tuple${arity}.html" -print -quit)"
    test -z "$(find target/doc/massively -type f -name "type.Tuple${arity}.html" -print -quit)"
done

for arity in {3..12}; do
    test -z "$(find target/doc/massively -type f -name "fn.flatten${arity}.html" -print -quit)"
done

algorithms=(
    adjacent_difference adjacent_find all_of any_of copy copy_where count_if equal
    exclusive_scan exclusive_scan_by_key fill find_first_of find_if flat_map gather gather_where
    inclusive_scan inclusive_scan_by_key is_partitioned is_sorted is_sorted_until
    lexicographical_compare lower_bound map max_element merge merge_by_key min_element
    minmax_element mismatch none_of partition reduce reduce_by_key remove_where
    replace_where reverse scatter scatter_reduce scatter_where set_difference
    set_intersection set_union sort sort_by_key radix_sort_by_key transform_where unique
    unique_by_key upper_bound
)

for algorithm in "${algorithms[@]}"; do
    test -f "target/doc/massively/vector/fn.${algorithm}.html"
    test ! -e "target/doc/massively/fn.${algorithm}.html"
done

scalar_value_algorithms=(
    exclusive_scan exclusive_scan_by_key fill reduce reduce_by_key replace_where scatter_reduce
)

for algorithm in "${scalar_value_algorithms[@]}"; do
    rg -q 'MVal' "target/doc/massively/vector/fn.${algorithm}.html"
done

mapfile -t executor_pages < <(find target/doc/massively -name 'struct.Executor.html')
test "${#executor_pages[@]}" -gt 0
! rg -q 'Scalar' "${executor_pages[@]}"
! rg -q 'MVal' "${executor_pages[@]}"
rg -q 'MIndex' "${executor_pages[@]}"
! rg -q 'fill_with' "target/doc/massively/trait.MIterMut.html"
! rg -q 'method\.(len|is_empty|capacity)' \
    "target/doc/massively/trait.MIter.html" \
    "target/doc/massively/trait.MIterMut.html" \
    "target/doc/massively/trait.MStorage.html"

test -f "target/doc/massively/seg/struct.Segmentation.html"
test ! -e "target/doc/massively/struct.Segmentation.html"
test ! -e "target/doc/massively/fn.segments_with_context.html"
test ! -e "target/doc/massively/seg/fn.segments_with_context.html"
test ! -e "target/doc/massively/seg/struct.SegmentContexts.html"
test ! -e "target/doc/massively/seg/fn.flat_map_segments.html"
test ! -e "target/doc/massively/seg/fn.adjacent_flat_map.html"
test ! -e "target/doc/massively/iteration/index.html"

operations=(BinaryPredicateOp ExpandOp PredicateOp ReductionOp UnaryOp)

for operation in "${operations[@]}"; do
    test -f "target/doc/massively/op/trait.${operation}.html"
    test ! -e "target/doc/massively/trait.${operation}.html"
done

internal_operations=(U32ToBool U32ToUsize)

for operation in "${internal_operations[@]}"; do
    test ! -e "target/doc/massively/op/struct.${operation}.html"
done
